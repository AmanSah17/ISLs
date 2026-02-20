import itertools
import json
import logging
import math
import os
import time
import gc
import sys
# 'resource' is Linux/macOS only — use psutil instead on all platforms.
try:
    import resource as _resource_mod
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from PLSN_Extraction.modules.clustering import CLIQUEClusterer


@dataclass
class TuningConfig:
    k_values: list[int]
    r_values: list[float]
    neighbor_modes: list[str]
    min_dense_points_values: list[int | None]
    min_port_points: int
    expected_ports_min: int | None = None
    expected_ports_max: int | None = None


class CLIQUEHyperparameterTuner:
    """
    Runs systematic hyperparameter sweeps for CLIQUE clustering.
    """

    def __init__(self, output_dir: str, require_cuda: bool = True, show_progress: bool = True):
        self.output_dir = output_dir
        self.require_cuda = require_cuda
        self.show_progress = show_progress
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _rss_mb() -> float:
        """Return current RSS memory in MiB (cross-platform)."""
        # Linux: fastest path via /proc/self/statm
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) >= 2:
                resident_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (resident_pages * page_size) / (1024.0 * 1024.0)
        except Exception:
            pass
        # psutil — works on Windows, macOS, Linux
        if _HAS_PSUTIL:
            try:
                return _psutil.Process().memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                pass
        # Unix resource module fallback (Linux/macOS)
        if _HAS_RESOURCE:
            try:
                rss = _resource_mod.getrusage(_resource_mod.RUSAGE_SELF).ru_maxrss
                if rss > 0:
                    return float(rss) / 1024.0
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _normalize_min_dense(value) -> str:
        if pd.isna(value):
            return "none"
        if value is None:
            return "none"
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "nan", ""}:
            return "none"
        return str(int(value))

    @classmethod
    def _combo_key(cls, k: int, r: float, mode: str, min_dense_points) -> tuple:
        return (int(k), float(r), str(mode), cls._normalize_min_dense(min_dense_points))

    def _port_count_alignment(
        self, effective_ports: int, expected_ports_min: int | None, expected_ports_max: int | None
    ) -> float:
        if expected_ports_min is None and expected_ports_max is None:
            return 0.5

        low = expected_ports_min if expected_ports_min is not None else expected_ports_max
        high = expected_ports_max if expected_ports_max is not None else expected_ports_min
        if low is None or high is None:
            return 0.5
        if low > high:
            low, high = high, low

        if low <= effective_ports <= high:
            return 1.0

        dist = min(abs(effective_ports - low), abs(effective_ports - high))
        width = max(10, high - low)
        return math.exp(-dist / width)

    def _score_trial(
        self,
        effective_ports: int,
        coverage_ratio: float,
        largest_cluster_share: float,
        runtime_sec: float,
        expected_ports_min: int | None,
        expected_ports_max: int | None,
    ) -> float:
        count_score = self._port_count_alignment(effective_ports, expected_ports_min, expected_ports_max)
        coverage_score = min(max(coverage_ratio, 0.0), 1.0)
        balance_score = 1.0 - min(max(largest_cluster_share, 0.0), 1.0)
        speed_score = 1.0 / (1.0 + max(runtime_sec, 0.0))
        return (0.50 * count_score) + (0.25 * coverage_score) + (0.20 * balance_score) + (0.05 * speed_score)

    @staticmethod
    def _pick_columns(df: pd.DataFrame) -> tuple[str, str]:
        lat_col = "LAT" if "LAT" in df.columns else "Latitude"
        lon_col = "LON" if "LON" in df.columns else "Longitude"
        return lat_col, lon_col

    @staticmethod
    def _pick_time_col(df: pd.DataFrame) -> str | None:
        for col in ["BASEDATETIME", "Timestamp", "DATETIME"]:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _compute_edge_metrics(clustered_df: pd.DataFrame) -> tuple[int, int]:
        if clustered_df.empty or "cluster_id" not in clustered_df.columns or "MMSI" not in clustered_df.columns:
            return 0, 0

        time_col = CLIQUEHyperparameterTuner._pick_time_col(clustered_df)
        work = clustered_df[clustered_df["cluster_id"] != -1].copy()
        if work.empty:
            return 0, 0

        if time_col:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work = work[work[time_col].notna()].copy()
            work.sort_values(["MMSI", time_col], inplace=True)
        else:
            work.sort_values(["MMSI"], inplace=True)

        work["prev_cluster"] = work.groupby("MMSI")["cluster_id"].shift(1)
        work = work[(work["prev_cluster"].isna()) | (work["prev_cluster"] != work["cluster_id"])].copy()
        work["next_cluster"] = work.groupby("MMSI")["cluster_id"].shift(-1)
        transitions = work[
            work["next_cluster"].notna() & (work["cluster_id"] != work["next_cluster"])
        ][["MMSI", "cluster_id", "next_cluster"]].copy()

        if transitions.empty:
            return 0, 0

        transitions["next_cluster"] = transitions["next_cluster"].astype(int)
        grouped = transitions.groupby(["cluster_id", "next_cluster"]).size()
        edge_count = int(grouped.shape[0])
        transition_count = int(grouped.sum())
        return edge_count, transition_count

    def run_sweep(
        self,
        stationary_df: pd.DataFrame,
        cfg: TuningConfig,
        checkpoint_path: str | None = None,
        resume: bool = False,
        memory_cap_mb: float | None = None,
    ) -> pd.DataFrame:
        result_columns = [
            "trial",
            "k",
            "r",
            "neighbor_mode",
            "min_dense_points",
            "runtime_sec",
            "total_points",
            "coverage_ratio",
            "noise_ratio",
            "n_clusters",
            "effective_ports",
            "largest_cluster_points",
            "largest_cluster_share",
            "mean_cluster_points",
            "median_cluster_points",
            "edge_count",
            "transition_count",
            "score",
            "rss_mb",
        ]
        lat_col, lon_col = self._pick_columns(stationary_df)
        cols = [c for c in stationary_df.columns if c in [lat_col, lon_col, "MMSI", "BASEDATETIME", "Timestamp"]]
        work = stationary_df[cols].copy()
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce", downcast="float")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce", downcast="float")
        work = work[work[lat_col].notna() & work[lon_col].notna()].copy()
        total_points = len(work)
        if total_points == 0:
            raise ValueError("stationary_df is empty. Cannot run tuning.")

        combos = list(
            itertools.product(
                cfg.k_values,
                cfg.r_values,
                cfg.neighbor_modes,
                cfg.min_dense_points_values,
            )
        )
        self.logger.info("Starting CLIQUE hyperparameter sweep with %d trials...", len(combos))

        prior_rows: list[dict] = []
        completed_keys = set()
        if checkpoint_path:
            if resume and os.path.exists(checkpoint_path):
                try:
                    old = pd.read_csv(checkpoint_path)
                except pd.errors.EmptyDataError:
                    old = pd.DataFrame(columns=result_columns)
                if not old.empty:
                    old["_md_norm"] = old["min_dense_points"].apply(self._normalize_min_dense)
                    old = old.drop_duplicates(subset=["k", "r", "neighbor_mode", "_md_norm"], keep="last").drop(columns=["_md_norm"])
                    prior_rows = old.to_dict(orient="records")
                    for _, row in old.iterrows():
                        completed_keys.add(
                            self._combo_key(
                                int(row["k"]),
                                float(row["r"]),
                                str(row["neighbor_mode"]),
                                row["min_dense_points"],
                            )
                        )
                self.logger.info(
                    "Resume enabled: loaded %d completed trials from %s.",
                    len(completed_keys),
                    checkpoint_path,
                )
            elif (not resume) and os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        pending: list[tuple[int, float, str, int | None, int]] = []
        for pos, (k, r, mode, min_dense_points) in enumerate(combos, start=1):
            key = self._combo_key(k, r, mode, min_dense_points)
            if key in completed_keys:
                continue
            pending.append((k, r, mode, min_dense_points, pos))

        rows: list[dict] = []
        iterator = pending
        if self.show_progress:
            iterator = tqdm(pending, total=len(pending), desc="CLIQUE tuning", unit="trial")

        for run_idx, (k, r, mode, min_dense_points, trial_pos) in enumerate(iterator, start=1):
            current_rss = self._rss_mb()
            if memory_cap_mb is not None and current_rss > float(memory_cap_mb):
                self.logger.warning(
                    "Stopping sweep due to memory cap. Current RSS %.1fMB exceeds cap %.1fMB.",
                    current_rss,
                    float(memory_cap_mb),
                )
                break
            t0 = time.perf_counter()

            clusterer = CLIQUEClusterer(
                k=int(k),
                density_threshold_r=float(r),
                min_dense_points=min_dense_points,
                neighbor_mode=mode,
                require_cuda=self.require_cuda,
            )
            clustered = clusterer.fit_predict(work)
            runtime_sec = time.perf_counter() - t0

            non_noise = clustered[clustered["cluster_id"] != -1]
            coverage_ratio = len(non_noise) / total_points
            noise_ratio = 1.0 - coverage_ratio

            if non_noise.empty:
                cluster_sizes = pd.Series(dtype=int)
                n_clusters = 0
                largest_cluster_points = 0
                effective_ports = 0
            else:
                cluster_sizes = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
                n_clusters = int(cluster_sizes.shape[0])
                largest_cluster_points = int(cluster_sizes.iloc[0])
                effective_ports = int((cluster_sizes >= cfg.min_port_points).sum())

            edge_count, transition_count = self._compute_edge_metrics(clustered)

            median_cluster_points = float(cluster_sizes.median()) if n_clusters > 0 else 0.0
            mean_cluster_points = float(cluster_sizes.mean()) if n_clusters > 0 else 0.0
            largest_cluster_share = (
                largest_cluster_points / len(non_noise) if len(non_noise) > 0 else 0.0
            )

            score = self._score_trial(
                effective_ports=effective_ports,
                coverage_ratio=coverage_ratio,
                largest_cluster_share=largest_cluster_share,
                runtime_sec=runtime_sec,
                expected_ports_min=cfg.expected_ports_min,
                expected_ports_max=cfg.expected_ports_max,
            )

            rows.append(
                {
                    "trial": int(trial_pos),
                    "k": int(k),
                    "r": float(r),
                    "neighbor_mode": mode,
                    "min_dense_points": min_dense_points,
                    "runtime_sec": runtime_sec,
                    "total_points": total_points,
                    "coverage_ratio": coverage_ratio,
                    "noise_ratio": noise_ratio,
                    "n_clusters": n_clusters,
                    "effective_ports": effective_ports,
                    "largest_cluster_points": largest_cluster_points,
                    "largest_cluster_share": largest_cluster_share,
                    "mean_cluster_points": mean_cluster_points,
                    "median_cluster_points": median_cluster_points,
                    "edge_count": edge_count,
                    "transition_count": transition_count,
                    "score": score,
                    "rss_mb": self._rss_mb(),
                }
            )
            self.logger.info(
                "Trial %d/%d | K=%d r=%.7f mode=%s min_dense=%s | clusters=%d effective_ports=%d score=%.4f runtime=%.2fs rss=%.1fMB",
                trial_pos,
                len(combos),
                int(k),
                float(r),
                mode,
                str(min_dense_points),
                n_clusters,
                effective_ports,
                score,
                runtime_sec,
                self._rss_mb(),
            )
            if checkpoint_path:
                row_df = pd.DataFrame([rows[-1]])
                row_df.to_csv(
                    checkpoint_path,
                    mode="a",
                    header=(not os.path.exists(checkpoint_path)),
                    index=False,
                )

            if run_idx % 10 == 0:
                self.logger.info(
                    "Progress checkpoint completed=%d pending=%d total=%d | current max RSS: %.1fMB",
                    len(prior_rows) + run_idx,
                    max(0, len(pending) - run_idx),
                    len(combos),
                    self._rss_mb(),
                )
            del clustered
            del non_noise
            del cluster_sizes
            gc.collect()

        all_rows = prior_rows + rows
        if not all_rows:
            self.logger.warning(
                "No trials were executed. This can happen if memory cap is too low. "
                "Returning empty results dataframe."
            )
            return pd.DataFrame(columns=result_columns)

        results = pd.DataFrame(all_rows)
        for col in result_columns:
            if col not in results.columns:
                results[col] = pd.NA

        # Score can be missing for malformed/legacy checkpoint rows; keep valid rows first.
        if "score" in results.columns:
            results["score"] = pd.to_numeric(results["score"], errors="coerce")
            results = results.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        else:
            results = results.reset_index(drop=True)
        return results

    def export_results(self, results_df: pd.DataFrame, prefix: str = "clique_tuning") -> tuple[str, str]:
        csv_path = os.path.join(self.output_dir, f"{prefix}_results.csv")
        summary_path = os.path.join(self.output_dir, f"{prefix}_summary.json")
        results_df.to_csv(csv_path, index=False)

        top = results_df.head(10).to_dict(orient="records")
        summary = {
            "generated_at_epoch": time.time(),
            "n_trials": int(len(results_df)),
            "best_trial": top[0] if top else None,
            "top_10": top,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Tuning results saved: %s", csv_path)
        self.logger.info("Tuning summary saved: %s", summary_path)
        return csv_path, summary_path
