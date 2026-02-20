from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from NLSN_Extraction import config
from NLSN_Extraction.modules.data_loader import AISDataLoader
from NLSN_Extraction.modules.nlsn_tuner import BestPLSNParams, NLSNGammaConfig, NLSNGammaTuner


class _DropNoisyWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Singular matrix. Likely caused by all points lying in an N-1 space." in msg:
            return False
        return True


def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.cuda").setLevel(logging.WARNING)
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
    logging.getLogger("pyogrio").setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropNoisyWarnings())


def parse_gamma_values(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        values.append(float(t))
    if not values:
        raise ValueError("At least one gamma value is required.")
    return values


def _normalize_neighbor_mode(value) -> str:
    mode = str(value).strip()
    if mode.endswith(".0"):
        mode = mode[:-2]
    if mode not in {"4", "8"}:
        raise ValueError(f"neighbor_mode must be 4 or 8, got: {value}")
    return mode


def _safe_min_dense(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"", "none", "null", "nan"}:
            return None
    return int(value)


def load_best_plsn_params(
    summary_json_path: str,
    results_csv_path: str,
    k_override: int | None,
    r_override: float | None,
    neighbor_override: str | None,
    min_dense_override: str | None,
) -> BestPLSNParams:
    if k_override is not None and r_override is not None and neighbor_override is not None:
        return BestPLSNParams(
            k=int(k_override),
            r=float(r_override),
            neighbor_mode=_normalize_neighbor_mode(neighbor_override),
            min_dense_points=_safe_min_dense(min_dense_override),
        )

    if summary_json_path and os.path.exists(summary_json_path):
        with open(summary_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("best_trial")
        if best:
            return BestPLSNParams(
                k=int(best["k"]),
                r=float(best["r"]),
                neighbor_mode=_normalize_neighbor_mode(best["neighbor_mode"]),
                min_dense_points=_safe_min_dense(best.get("min_dense_points")),
            )

    if results_csv_path and os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df = df[df["score"].notna()].copy()
            if not df.empty:
                df = df.sort_values("score", ascending=False).reset_index(drop=True)
        if not df.empty:
            row = df.iloc[0].to_dict()
            return BestPLSNParams(
                k=int(row["k"]),
                r=float(row["r"]),
                neighbor_mode=_normalize_neighbor_mode(row["neighbor_mode"]),
                min_dense_points=_safe_min_dense(row.get("min_dense_points")),
            )

    return BestPLSNParams(
        k=int(config.CLIQUE_GRID_DIVISIONS_K),
        r=float(config.CLIQUE_DENSITY_THRESHOLD_R),
        neighbor_mode=_normalize_neighbor_mode(config.CLIQUE_NEIGHBOR_MODE),
        min_dense_points=_safe_min_dense(config.CLIQUE_MIN_DENSE_POINTS),
    )


def parse_args():
    default_full_data = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results/data/jan2025_full_minimal.parquet"
    if not os.path.exists(default_full_data):
        default_full_data = config.FULL_DATA_PATH

    default_plsn_summary = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results/tuning/expanded_hyperparam_sweep_summary.json"
    default_plsn_results = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results/tuning/expanded_hyperparam_sweep_results.csv"
    default_output_dir = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results/nlsn_gamma_sweep"

    parser = argparse.ArgumentParser(description="NLSN gamma hyperparameter sweep using best PLSN credentials.")
    parser.add_argument("--full-data-path", type=str, default=default_full_data)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--plsn-summary-json", type=str, default=default_plsn_summary)
    parser.add_argument("--plsn-results-csv", type=str, default=default_plsn_results)
    parser.add_argument(
        "--gamma-values",
        type=str,
        default="0.00005,0.00008,0.0001,0.00015,0.0002,0.0003,0.0004",
        help="Comma-separated DP gamma candidates.",
    )
    parser.add_argument("--min-trajectory-points", type=int, default=5)
    parser.add_argument("--max-time-gap-minutes", type=float, default=720.0)
    parser.add_argument("--w1", type=float, default=1.0, help="LD weight for compression rate Dr.")
    parser.add_argument("--w2", type=float, default=1.0, help="LD weight for distance similarity Dl.")
    parser.add_argument("--alpha-shape", type=float, default=config.ALPHA_SHAPE_PARAMETER)
    parser.add_argument("--expected-nodes-min", type=int, default=None)
    parser.add_argument("--expected-nodes-max", type=int, default=None)
    parser.add_argument("--map-sample-size", type=int, default=120000)
    parser.add_argument("--require-cuda", action="store_true", default=False)
    parser.add_argument("--no-cuda", dest="require_cuda", action="store_false")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=5_000_000,
        help="Safety cap for rows loaded into memory (0 disables cap).",
    )
    parser.add_argument(
        "--sample-mmsi-count",
        type=int,
        default=0,
        help="If >0, load only this many MMSIs (memory-safe sampling).",
    )

    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--neighbor-mode", type=str, default=None)
    parser.add_argument("--min-dense-points", type=str, default=None)
    return parser.parse_args()


def _estimate_parquet_rows(path: str) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _load_input_df(
    data_path: str,
    max_input_rows: int,
    sample_mmsi_count: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    cols = ["MMSI", "BASEDATETIME", "LAT", "LON"]
    max_rows = int(max_input_rows) if int(max_input_rows) > 0 else 0

    if duckdb is not None and (sample_mmsi_count > 0 or max_rows > 0):
        safe_path = data_path.replace("'", "''")

        if sample_mmsi_count > 0:
            logger.info(
                "Loading sampled MMSI subset via DuckDB (sample_mmsi_count=%d, max_input_rows=%s).",
                int(sample_mmsi_count),
                str(max_rows if max_rows > 0 else "none"),
            )
            limit_clause = f"LIMIT {max_rows}" if max_rows > 0 else ""
            query = f"""
WITH sampled_mmsi AS (
    SELECT CAST(MMSI AS BIGINT) AS MMSI
    FROM read_parquet('{safe_path}')
    WHERE MMSI IS NOT NULL
    GROUP BY 1
    ORDER BY 1
    LIMIT {int(sample_mmsi_count)}
)
SELECT
    CAST(t.MMSI AS BIGINT) AS MMSI,
    CAST(t.BASEDATETIME AS TIMESTAMP) AS BASEDATETIME,
    CAST(t.LAT AS DOUBLE) AS LAT,
    CAST(t.LON AS DOUBLE) AS LON
FROM read_parquet('{safe_path}') AS t
INNER JOIN sampled_mmsi s
    ON CAST(t.MMSI AS BIGINT) = s.MMSI
WHERE t.MMSI IS NOT NULL
  AND t.BASEDATETIME IS NOT NULL
  AND t.LAT BETWEEN -90 AND 90
  AND t.LON BETWEEN -180 AND 180
ORDER BY CAST(t.MMSI AS BIGINT), CAST(t.BASEDATETIME AS TIMESTAMP)
{limit_clause}
"""
        else:
            logger.info(
                "Loading capped subset via DuckDB LIMIT (max_input_rows=%d).",
                max_rows,
            )
            query = f"""
SELECT
    CAST(MMSI AS BIGINT) AS MMSI,
    CAST(BASEDATETIME AS TIMESTAMP) AS BASEDATETIME,
    CAST(LAT AS DOUBLE) AS LAT,
    CAST(LON AS DOUBLE) AS LON
FROM read_parquet('{safe_path}')
WHERE MMSI IS NOT NULL
  AND BASEDATETIME IS NOT NULL
  AND LAT BETWEEN -90 AND 90
  AND LON BETWEEN -180 AND 180
LIMIT {max_rows}
"""

        con = duckdb.connect()
        df = con.execute(query).fetch_df()
        con.close()
    elif sample_mmsi_count > 0 and duckdb is None:
        logger.warning("duckdb is unavailable, falling back to pandas loader without MMSI sampling.")
        logger.info("Loading minimal columns only: %s", ", ".join(cols))
        df = pd.read_parquet(data_path, columns=cols)
        if max_rows > 0 and len(df) > max_rows:
            df = df.head(max_rows).copy()
    elif duckdb is None:
        logger.info("Loading minimal columns only: %s", ", ".join(cols))
        df = pd.read_parquet(data_path, columns=cols)
        if max_rows > 0 and len(df) > max_rows:
            df = df.head(max_rows).copy()
    else:
        # duckdb available but no caps/sampling requested: explicit full load
        logger.info(
            "Loading full dataset with DuckDB (no row cap / no MMSI sampling). This can be memory-intensive."
        )
        con = duckdb.connect()
        query = f"""
SELECT
    CAST(MMSI AS BIGINT) AS MMSI,
    CAST(BASEDATETIME AS TIMESTAMP) AS BASEDATETIME,
    CAST(LAT AS DOUBLE) AS LAT,
    CAST(LON AS DOUBLE) AS LON
FROM read_parquet('{data_path.replace("'", "''")}')
WHERE MMSI IS NOT NULL
  AND BASEDATETIME IS NOT NULL
  AND LAT BETWEEN -90 AND 90
  AND LON BETWEEN -180 AND 180
"""
        df = con.execute(query).fetch_df()
        con.close()

    # Downcast to reduce memory footprint.
    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce", downcast="integer")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce", downcast="float")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce", downcast="float")
    df["BASEDATETIME"] = pd.to_datetime(df["BASEDATETIME"], errors="coerce")
    df = df[df["MMSI"].notna() & df["BASEDATETIME"].notna()].copy()
    logger.info("Loaded dataframe rows=%d columns=%s", len(df), list(df.columns))
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "nlsn_gamma_sweep.log")
    setup_logging(log_path)
    logger = logging.getLogger("NLSN-Gamma-Sweep")

    gamma_values = parse_gamma_values(args.gamma_values)
    plsn_params = load_best_plsn_params(
        summary_json_path=args.plsn_summary_json,
        results_csv_path=args.plsn_results_csv,
        k_override=args.k,
        r_override=args.r,
        neighbor_override=args.neighbor_mode,
        min_dense_override=args.min_dense_points,
    )
    logger.info(
        "Using best PLSN credentials: k=%d r=%.7f neighbor_mode=%s min_dense_points=%s",
        plsn_params.k,
        plsn_params.r,
        plsn_params.neighbor_mode,
        str(plsn_params.min_dense_points),
    )
    logger.info("Gamma candidates: %s", ", ".join([f"{g:g}" for g in gamma_values]))
    estimated_rows = _estimate_parquet_rows(args.full_data_path)
    if estimated_rows is not None:
        logger.info("Input parquet estimated rows: %d", estimated_rows)
    if estimated_rows is not None and estimated_rows > 20_000_000:
        logger.warning(
            "Very large parquet detected (%d rows). "
            "Use --sample-mmsi-count and/or --max-input-rows to avoid IDE or system OOM.",
            estimated_rows,
        )

    if args.max_input_rows <= 0 and estimated_rows is not None and estimated_rows > 20_000_000:
        logger.warning(
            "Auto-enabling safety cap max_input_rows=5000000 for large parquet. "
            "Set --max-input-rows 0 only if your RAM can handle full load."
        )
        args.max_input_rows = 5_000_000

    # Keep AISDataLoader import path untouched for project compatibility,
    # but use memory-safe loader here for NLSN runs on large monthly data.
    _ = AISDataLoader  # no-op reference
    full_df = _load_input_df(
        data_path=args.full_data_path,
        max_input_rows=int(args.max_input_rows),
        sample_mmsi_count=int(args.sample_mmsi_count),
        logger=logger,
    )
    cfg = NLSNGammaConfig(
        gamma_values=gamma_values,
        min_trajectory_points=int(args.min_trajectory_points),
        max_time_gap_minutes=float(args.max_time_gap_minutes),
        alpha_shape=float(args.alpha_shape),
        w1=float(args.w1),
        w2=float(args.w2),
        expected_nodes_min=args.expected_nodes_min,
        expected_nodes_max=args.expected_nodes_max,
        map_sample_size=int(args.map_sample_size),
    )

    tuner = NLSNGammaTuner(
        output_dir=args.output_dir,
        require_cuda=bool(args.require_cuda),
        show_progress=not bool(args.no_progress),
    )
    results_df = tuner.run_sweep(full_df=full_df, plsn_params=plsn_params, cfg=cfg)

    if results_df.empty:
        logger.warning("No NLSN results were produced.")
        return

    best = results_df.iloc[0].to_dict()
    final_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "full_data_path": args.full_data_path,
        "output_dir": args.output_dir,
        "plsn_params": {
            "k": plsn_params.k,
            "r": plsn_params.r,
            "neighbor_mode": plsn_params.neighbor_mode,
            "min_dense_points": plsn_params.min_dense_points,
        },
        "gamma_values": gamma_values,
        "best_trial": best,
        "results_csv": os.path.join(args.output_dir, "nlsn_gamma_sweep_results.csv"),
        "results_summary_json": os.path.join(args.output_dir, "nlsn_gamma_sweep_summary.json"),
        "comparison_html": os.path.join(args.output_dir, "nlsn_gamma_comparison.html"),
    }
    summary_path = os.path.join(args.output_dir, "nlsn_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    logger.info("Run complete. Best gamma=%.6g score=%.4f nodes=%d edges=%d transitions=%d",
                float(best["gamma"]), float(best["score"]), int(best["nodes"]), int(best["edges"]), int(best["transitions"]))
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
