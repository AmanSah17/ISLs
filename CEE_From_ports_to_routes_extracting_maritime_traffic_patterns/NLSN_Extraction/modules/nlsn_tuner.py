from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import resource  # Unix
except Exception:  # pragma: no cover
    resource = None

from NLSN_Extraction.modules.adaptive_dp import AdaptiveDPOptimizer
from NLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
from NLSN_Extraction.modules.clustering import CLIQUEClusterer
from NLSN_Extraction.modules.nlsn_generator import NLSNGenerator
from NLSN_Extraction.modules.visualizer import PLSNVisualizer


@dataclass
class BestPLSNParams:
    k: int
    r: float
    neighbor_mode: str
    min_dense_points: int | None = None


@dataclass
class NLSNGammaConfig:
    gamma_values: list[float]
    min_trajectory_points: int = 5
    max_time_gap_minutes: float = 720.0
    alpha_shape: float = 0.01
    w1: float = 1.0
    w2: float = 1.0
    expected_nodes_min: int | None = None
    expected_nodes_max: int | None = None
    map_sample_size: int = 120000


class NLSNGammaTuner:
    """
    Build NLSN by:
    1) Adaptive-DP feature extraction (fixed gamma per trial)
    2) CLIQUE clustering using best PLSN credentials
    3) NLSN edge extraction from cluster transitions
    4) Gamma sweep ranking and comparison artifacts
    """

    def __init__(self, output_dir: str, require_cuda: bool = True, show_progress: bool = True):
        self.output_dir = output_dir
        self.require_cuda = bool(require_cuda)
        self.show_progress = bool(show_progress)
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _rss_mb() -> float:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) >= 2:
                resident_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (resident_pages * page_size) / (1024.0 * 1024.0)
        except Exception:
            pass
        if resource is not None:
            try:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return float(rss) / 1024.0 if rss > 0 else 0.0
            except Exception:
                pass
        try:
            import psutil  # type: ignore

            rss = psutil.Process(os.getpid()).memory_info().rss
            return float(rss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise KeyError(f"Required column not found. Tried: {candidates}")
        return None

    @staticmethod
    def _gamma_token(gamma: float) -> str:
        return f"{gamma:g}".replace(".", "p")

    @staticmethod
    def _node_alignment(effective_nodes: int, expected_min: int | None, expected_max: int | None) -> float:
        if expected_min is None and expected_max is None:
            return 0.5
        low = expected_min if expected_min is not None else expected_max
        high = expected_max if expected_max is not None else expected_min
        if low is None or high is None:
            return 0.5
        if low > high:
            low, high = high, low
        if low <= effective_nodes <= high:
            return 1.0
        dist = min(abs(effective_nodes - low), abs(effective_nodes - high))
        width = max(10, high - low)
        return math.exp(-dist / width)

    def _prepare_trajectories(
        self,
        df: pd.DataFrame,
        cfg: NLSNGammaConfig,
    ) -> tuple[list[dict], pd.DataFrame]:
        lat_col = self._pick_column(df, ["LAT", "Latitude"])
        lon_col = self._pick_column(df, ["LON", "Longitude"])
        mmsi_col = self._pick_column(df, ["MMSI"])
        time_col = self._pick_column(df, ["BASEDATETIME", "Timestamp", "DATETIME"], required=False)

        cols = [mmsi_col, lat_col, lon_col]
        if time_col:
            cols.append(time_col)
        work = df[cols].copy()

        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
        work = work[
            work[mmsi_col].notna()
            & work[lat_col].between(-90, 90)
            & work[lon_col].between(-180, 180)
        ].copy()

        if time_col:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work = work[work[time_col].notna()].copy()
            work.sort_values([mmsi_col, time_col], inplace=True)
            gap = work.groupby(mmsi_col)[time_col].diff().dt.total_seconds().div(60.0)
            new_track = gap.isna() | (gap > float(cfg.max_time_gap_minutes))
            track_seq = new_track.groupby(work[mmsi_col]).cumsum().astype(int)
            work["trajectory_id"] = work[mmsi_col].astype(str) + "_" + track_seq.astype(str)
        else:
            work.sort_values([mmsi_col], inplace=True)
            work["trajectory_id"] = work[mmsi_col].astype(str) + "_0"

        work["point_order"] = work.groupby("trajectory_id").cumcount()

        group_sizes = work.groupby("trajectory_id").size()
        valid_tracks = set(group_sizes[group_sizes >= int(cfg.min_trajectory_points)].index.tolist())
        work = work[work["trajectory_id"].isin(valid_tracks)].copy()
        if work.empty:
            raise ValueError(
                "No trajectories left after minimum-point filtering. "
                "Lower --min-trajectory-points or check input data."
            )

        segments: list[dict] = []
        grouped = work.groupby("trajectory_id", sort=False)
        iterator = grouped
        if self.show_progress:
            iterator = tqdm(grouped, total=len(grouped), desc="Prepare trajectories", unit="traj")

        for trajectory_id, grp in iterator:
            grp = grp.sort_values("point_order")
            points = grp[[lon_col, lat_col]].to_numpy(dtype=np.float64)
            if points.shape[0] < int(cfg.min_trajectory_points):
                continue
            seg = {
                "trajectory_id": str(trajectory_id),
                "mmsi": int(grp[mmsi_col].iloc[0]),
                "points": points,
            }
            if time_col:
                seg["times"] = grp[time_col].to_numpy()
            else:
                seg["times"] = None
            segments.append(seg)

        map_context = work[[lat_col, lon_col]].rename(columns={lat_col: "LAT", lon_col: "LON"})
        if len(map_context) > 400000:
            map_context = map_context.sample(n=400000, random_state=42)

        self.logger.info(
            "Prepared %d trajectories (%d points) for NLSN extraction.",
            len(segments),
            int(sum(seg["points"].shape[0] for seg in segments)),
        )
        return segments, map_context

    def _extract_feature_points_for_gamma(
        self,
        segments: list[dict],
        gamma: float,
        optimizer: AdaptiveDPOptimizer,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_rows: list[dict] = []
        metric_rows: list[dict] = []

        iterator = segments
        if self.show_progress:
            iterator = tqdm(segments, total=len(segments), desc=f"DP gamma={gamma:g}", unit="traj")

        for seg in iterator:
            result = optimizer.compress_with_gamma(seg["points"], gamma=gamma)
            keep = result.keep_indices
            metrics = result.metrics
            metric_rows.append(
                {
                    "trajectory_id": seg["trajectory_id"],
                    "MMSI": seg["mmsi"],
                    "gamma": gamma,
                    "original_points": metrics.original_points,
                    "compressed_points": metrics.compressed_points,
                    "compression_rate_dr": metrics.compression_rate_dr,
                    "distance_similarity_dl": metrics.distance_similarity_dl,
                    "distance_loss": metrics.distance_loss,
                    "original_length": metrics.original_length,
                    "compressed_length": metrics.compressed_length,
                    "ld_score": metrics.ld_score,
                }
            )

            times = seg["times"]
            for order, idx in enumerate(keep):
                row = {
                    "MMSI": seg["mmsi"],
                    "trajectory_id": seg["trajectory_id"],
                    "feature_order": int(order),
                    "feature_idx_in_traj": int(idx),
                    "LON": float(seg["points"][idx, 0]),
                    "LAT": float(seg["points"][idx, 1]),
                    "gamma": float(gamma),
                }
                if times is not None:
                    row["BASEDATETIME"] = pd.Timestamp(times[idx])
                feature_rows.append(row)

        feature_df = pd.DataFrame(feature_rows)
        if not feature_df.empty and "BASEDATETIME" in feature_df.columns:
            feature_df["BASEDATETIME"] = pd.to_datetime(feature_df["BASEDATETIME"], errors="coerce")

        metric_df = pd.DataFrame(metric_rows)
        return feature_df, metric_df

    def _score_trial(
        self,
        mean_ld: float,
        cluster_coverage: float,
        largest_cluster_share: float,
        n_nodes: int,
        edge_count: int,
        transition_count: int,
        cfg: NLSNGammaConfig,
    ) -> float:
        balance = 1.0 - min(max(largest_cluster_share, 0.0), 1.0)
        node_alignment = self._node_alignment(n_nodes, cfg.expected_nodes_min, cfg.expected_nodes_max)
        density = 0.0
        if n_nodes > 1:
            density = edge_count / max(1, n_nodes * (n_nodes - 1))
        density_scaled = 1.0 - math.exp(-25.0 * density)
        transitions_per_edge = transition_count / max(1, edge_count)
        transition_scaled = min(1.0, transitions_per_edge / 3.0)

        return (
            0.35 * mean_ld
            + 0.20 * cluster_coverage
            + 0.15 * balance
            + 0.15 * node_alignment
            + 0.10 * density_scaled
            + 0.05 * transition_scaled
        )

    def run_sweep(
        self,
        full_df: pd.DataFrame,
        plsn_params: BestPLSNParams,
        cfg: NLSNGammaConfig,
    ) -> pd.DataFrame:
        if not cfg.gamma_values:
            raise ValueError("gamma_values cannot be empty.")

        segments, map_context = self._prepare_trajectories(full_df, cfg)
        optimizer = AdaptiveDPOptimizer(w1=cfg.w1, w2=cfg.w2)

        rows: list[dict] = []
        trial_summaries: list[dict] = []

        sweep_iter = cfg.gamma_values
        if self.show_progress:
            sweep_iter = tqdm(cfg.gamma_values, total=len(cfg.gamma_values), desc="NLSN gamma sweep", unit="gamma")

        for trial_idx, gamma in enumerate(sweep_iter, start=1):
            trial_t0 = time.perf_counter()
            gamma = float(gamma)
            gamma_key = self._gamma_token(gamma)
            trial_dir = os.path.join(self.output_dir, f"gamma_{gamma_key}")
            os.makedirs(trial_dir, exist_ok=True)

            feature_df, seg_metrics_df = self._extract_feature_points_for_gamma(segments, gamma, optimizer)
            generator = NLSNGenerator(output_dir=trial_dir)
            feature_path = generator.export_feature_points(feature_df)

            if feature_df.empty:
                clustered_df = feature_df.copy()
                clustered_df["cluster_id"] = []
                boundaries = []
                nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "feature_points", "area_deg2"])
                edges_df = pd.DataFrame(columns=["source_port", "target_port", "transition_count", "unique_vessels"])
                nodes_df.to_csv(os.path.join(trial_dir, "nodes.csv"), index=False)
                edges_df.to_csv(os.path.join(trial_dir, "edges.csv"), index=False)
            else:
                clusterer = CLIQUEClusterer(
                    k=int(plsn_params.k),
                    density_threshold_r=float(plsn_params.r),
                    min_dense_points=plsn_params.min_dense_points,
                    neighbor_mode=str(plsn_params.neighbor_mode),
                    require_cuda=self.require_cuda,
                )
                clustered_df = clusterer.fit_predict(feature_df)
                boundaries = BoundaryExtractor(alpha=cfg.alpha_shape).extract_boundaries(clustered_df)
                nodes_df = generator.export_nodes_and_boundaries(boundaries, clustered_df)
                valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
                edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)

            total_original_points = int(seg_metrics_df["original_points"].sum()) if not seg_metrics_df.empty else 0
            total_compressed_points = int(seg_metrics_df["compressed_points"].sum()) if not seg_metrics_df.empty else 0
            orig_len_total = float(seg_metrics_df["original_length"].sum()) if not seg_metrics_df.empty else 0.0
            comp_len_total = float(seg_metrics_df["compressed_length"].sum()) if not seg_metrics_df.empty else 0.0

            dr_global = 1.0 - (total_compressed_points / max(1, total_original_points))
            dl_global = 1.0 - (abs(comp_len_total - orig_len_total) / max(orig_len_total, 1e-12))
            dl_global = float(np.clip(dl_global, 0.0, 1.0))
            ld_global = float((cfg.w1 * dr_global) + (cfg.w2 * dl_global))

            n_feature_points = int(len(feature_df))
            non_noise_count = int((clustered_df["cluster_id"] != -1).sum()) if "cluster_id" in clustered_df.columns else 0
            cluster_coverage = non_noise_count / max(1, n_feature_points)

            non_noise = clustered_df[clustered_df["cluster_id"] != -1] if "cluster_id" in clustered_df.columns else clustered_df
            if non_noise.empty:
                n_clusters = 0
                largest_cluster_points = 0
                largest_cluster_share = 0.0
            else:
                sizes = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
                n_clusters = int(len(sizes))
                largest_cluster_points = int(sizes.iloc[0])
                largest_cluster_share = largest_cluster_points / max(1, non_noise_count)

            n_nodes = int(len(nodes_df))
            edge_count = int(len(edges_df))
            transition_count = int(pd.to_numeric(edges_df["transition_count"], errors="coerce").fillna(0).sum()) if "transition_count" in edges_df.columns else 0

            trial_score = self._score_trial(
                mean_ld=ld_global,
                cluster_coverage=cluster_coverage,
                largest_cluster_share=largest_cluster_share,
                n_nodes=n_nodes,
                edge_count=edge_count,
                transition_count=transition_count,
                cfg=cfg,
            )

            nodes_for_vis = nodes_df.rename(columns={"feature_points": "stationary_points"})
            map_path = os.path.join(trial_dir, "nlsn_map.html")
            PLSNVisualizer(output_file=map_path, sample_size=cfg.map_sample_size).generate_plsn_dashboard(
                map_context,
                clustered_df,
                boundaries,
                nodes_for_vis,
                edges_df,
            )

            runtime_sec = time.perf_counter() - trial_t0
            row = {
                "trial": int(trial_idx),
                "gamma": gamma,
                "k": int(plsn_params.k),
                "r": float(plsn_params.r),
                "neighbor_mode": str(plsn_params.neighbor_mode),
                "min_dense_points": plsn_params.min_dense_points,
                "trajectories": int(len(segments)),
                "original_points": total_original_points,
                "feature_points": total_compressed_points,
                "compression_rate_dr": dr_global,
                "distance_similarity_dl": dl_global,
                "ld_score": ld_global,
                "cluster_coverage": cluster_coverage,
                "n_clusters": n_clusters,
                "largest_cluster_points": largest_cluster_points,
                "largest_cluster_share": largest_cluster_share,
                "nodes": n_nodes,
                "edges": edge_count,
                "transitions": transition_count,
                "score": trial_score,
                "runtime_sec": runtime_sec,
                "rss_mb": self._rss_mb(),
                "trial_dir": trial_dir,
                "feature_points_csv": feature_path,
                "map_html": map_path,
            }
            rows.append(row)
            trial_summary = dict(row)
            trial_summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
            trial_summary["plsn_params"] = {
                "k": int(plsn_params.k),
                "r": float(plsn_params.r),
                "neighbor_mode": str(plsn_params.neighbor_mode),
                "min_dense_points": plsn_params.min_dense_points,
            }
            trial_summaries.append(trial_summary)
            with open(os.path.join(trial_dir, "nlsn_trial_summary.json"), "w", encoding="utf-8") as f:
                json.dump(trial_summary, f, indent=2)

            self.logger.info(
                "Gamma %.6g | nodes=%d edges=%d transitions=%d LD=%.4f coverage=%.4f score=%.4f runtime=%.2fs",
                gamma,
                n_nodes,
                edge_count,
                transition_count,
                ld_global,
                cluster_coverage,
                trial_score,
                runtime_sec,
            )

            del feature_df
            del seg_metrics_df
            del clustered_df
            del non_noise
            del nodes_df
            del edges_df
            gc.collect()

        results_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        results_csv = os.path.join(self.output_dir, "nlsn_gamma_sweep_results.csv")
        results_df.to_csv(results_csv, index=False)

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_trials": int(len(results_df)),
            "best_trial": results_df.iloc[0].to_dict() if not results_df.empty else None,
            "all_trials": trial_summaries,
            "results_csv": results_csv,
        }
        summary_path = os.path.join(self.output_dir, "nlsn_gamma_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._build_comparison_html(results_df)
        self.logger.info("NLSN gamma sweep complete: %s", results_csv)
        return results_df

    def _build_comparison_html(self, results_df: pd.DataFrame):
        html_path = os.path.join(self.output_dir, "nlsn_gamma_comparison.html")
        if results_df.empty:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h3>No NLSN results available.</h3></body></html>")
            return

        methods = []
        for _, row in results_df.iterrows():
            methods.append(
                {
                    "label": f"gamma={row['gamma']:.6g} | score={row['score']:.4f}",
                    "gamma": float(row["gamma"]),
                    "score": float(row["score"]),
                    "nodes": int(row["nodes"]),
                    "edges": int(row["edges"]),
                    "transitions": int(row["transitions"]),
                    "compression_rate_dr": float(row["compression_rate_dr"]),
                    "distance_similarity_dl": float(row["distance_similarity_dl"]),
                    "ld_score": float(row["ld_score"]),
                    "cluster_coverage": float(row["cluster_coverage"]),
                    "map_rel": os.path.relpath(row["map_html"], self.output_dir),
                }
            )

        rows_html = "".join(
            [
                "<tr>"
                f"<td>{m['gamma']:.6g}</td>"
                f"<td>{m['score']:.4f}</td>"
                f"<td>{m['nodes']}</td>"
                f"<td>{m['edges']}</td>"
                f"<td>{m['transitions']}</td>"
                f"<td>{m['compression_rate_dr']:.4f}</td>"
                f"<td>{m['distance_similarity_dl']:.4f}</td>"
                f"<td>{m['ld_score']:.4f}</td>"
                f"<td>{m['cluster_coverage']:.4f}</td>"
                f"<td><a href='{m['map_rel']}' target='_blank'>Open Map</a></td>"
                "</tr>"
                for m in methods
            ]
        )

        default_map = methods[0]["map_rel"]
        methods_json = json.dumps(methods)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NLSN Gamma Sweep</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, sans-serif; background: #f8fafc; color: #111827; }}
    .wrap {{ padding: 16px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    iframe {{ width: 100%; height: 720px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; }}
    .scroll {{ max-height: 380px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
    .row {{ display: grid; grid-template-columns: 1fr 2fr; gap: 12px; }}
    @media (max-width: 1000px) {{ .row {{ grid-template-columns: 1fr; }} iframe {{ height: 520px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>NLSN Gamma Hyperparameter Comparison</h2>
    <p>Generated: {datetime.now().isoformat(timespec='seconds')}</p>
    <div class="card">
      <div class="row">
        <div>
          <label for="map-select"><b>Select gamma map:</b></label>
          <select id="map-select" style="width:100%;padding:8px;"></select>
          <div class="scroll">
            <table>
              <thead>
                <tr>
                  <th>Gamma</th><th>Score</th><th>Nodes</th><th>Edges</th><th>Transitions</th>
                  <th>Dr</th><th>Dl</th><th>LD</th><th>Coverage</th><th>Map</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
        </div>
        <div>
          <iframe id="map-frame" src="{default_map}"></iframe>
        </div>
      </div>
    </div>
  </div>
  <script>
    const methods = {methods_json};
    const select = document.getElementById('map-select');
    const frame = document.getElementById('map-frame');
    methods.forEach((m, idx) => {{
      const opt = document.createElement('option');
      opt.value = m.map_rel;
      opt.textContent = `${{idx + 1}}. gamma=${{m.gamma}} (score=${{m.score.toFixed(4)}})`;
      select.appendChild(opt);
    }});
    select.addEventListener('change', () => {{
      frame.src = select.value;
    }});
  </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
