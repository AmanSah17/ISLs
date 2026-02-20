"""
run_plsn_full.py — Single end-to-end PLSN pipeline entry point.

Workflow
────────
1. Load & validate AIS Parquet
2. Preprocess → stationary points (anchor/moored)
3. Extended CLIQUE hyperparameter sweep (saved to results/tuning/)
4. Re-run top-N configs → full pipeline (cluster → boundaries → PLSN network → map)
   Output saved to results/maps/top_N_<ts>/<method_name>/
5. Build enhanced comparison dashboard → results/maps/plsn_comparison_<ts>.html
6. Optionally run best config through entire pipeline one more time
   and save named artifacts → results/runs/<ts>/

Usage
────────
  python run_plsn_full.py                          # uses config.py defaults
  python run_plsn_full.py --no-cuda               # CPU-only
  python run_plsn_full.py --resume-from-checkpoint
  python run_plsn_full.py --top-maps 10 --memory-cap-mb 8192
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from PLSN_Extraction import config
from PLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
from PLSN_Extraction.modules.clustering import CLIQUEClusterer
from PLSN_Extraction.modules.data_loader import AISDataLoader
from PLSN_Extraction.modules.hyperparameter_tuner import CLIQUEHyperparameterTuner, TuningConfig
from PLSN_Extraction.modules.network_generator import PLSNGenerator
from PLSN_Extraction.modules.preprocessor import AISPreprocessor
from PLSN_Extraction.modules.visualizer import PLSNVisualizer
from run_hyperparam_comparison import build_comparison_html, method_id


# ── Logging ────────────────────────────────────────────────────────────────────
class _DropNoisyWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Singular matrix" not in record.getMessage()


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    for lib in ("numba", "numba.cuda", "numba.cuda.cudadrv.driver", "pyogrio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropNoisyWarnings())


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def run_pipeline_for_params(
    stationary_df: pd.DataFrame,
    map_full_df:   pd.DataFrame,
    output_dir:    str,
    k:             int,
    r:             float,
    neighbor_mode: str,
    min_dense_points: int | None,
    use_cuda:      bool,
    trial_score:   float | None = None,
    logger:        logging.Logger | None = None,
) -> dict:
    """
    Run the full PLSN pipeline for one parameter configuration and return a
    summary dict with paths to generated artefacts.
    """
    log = logger or logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    # Cluster
    clusterer = CLIQUEClusterer(
        k=k,
        density_threshold_r=r,
        min_dense_points=min_dense_points,
        neighbor_mode=neighbor_mode,
        require_cuda=use_cuda,
    )
    clustered_df = clusterer.fit_predict(stationary_df)
    log.info("  Clusters found: %d", clustered_df["cluster_id"].nunique() - (1 if -1 in clustered_df["cluster_id"].values else 0))

    # Boundaries
    boundaries = BoundaryExtractor(alpha=config.ALPHA_SHAPE_PARAMETER).extract_boundaries(clustered_df)

    # Network artefacts
    generator = PLSNGenerator(output_dir=output_dir)
    nodes_df  = generator.export_nodes_and_boundaries(boundaries, clustered_df=clustered_df)
    if nodes_df is None or nodes_df.empty:
        nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "area_deg2", "stationary_points"])
    valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
    edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)

    # Interactive HTML map
    params_meta = {"k": k, "r": r, "neighbor_mode": neighbor_mode, "min_dense_points": min_dense_points}
    map_path = os.path.join(output_dir, "plsn_map.html")
    PLSNVisualizer(
        output_file=map_path,
        sample_size=config.VISUALIZATION_SAMPLE_SIZE,
        edge_min_width=config.EDGE_MIN_WIDTH,
        edge_max_width=config.EDGE_MAX_WIDTH,
        boundary_weight=config.BOUNDARY_WEIGHT,
        node_min_radius=config.NODE_MIN_RADIUS,
        node_max_radius=config.NODE_MAX_RADIUS,
    ).generate_plsn_dashboard(
        map_full_df, clustered_df, boundaries, nodes_df, edges_df,
        params_meta=params_meta, trial_score=trial_score,
    )
    log.info("  Map saved: %s", map_path)

    return {
        "nodes_csv":    os.path.join(output_dir, "nodes.csv"),
        "edges_csv":    os.path.join(output_dir, "edges.csv"),
        "boundaries":   os.path.join(output_dir, "boundaries.geojson"),
        "map_html":     map_path,
        "n_ports":      len(nodes_df),
        "n_edges":      len(edges_df),
        "n_transitions": int(edges_df["transition_count"].sum()) if not edges_df.empty else 0,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end PLSN pipeline: hyperparameter sweep → maps → comparison dashboard."
    )
    parser.add_argument("--data-path",    type=str,   default=config.DATA_FILE_PATH,
                        help="Path to processed AIS Parquet file.")
    parser.add_argument("--results-dir",  type=str,   default=config.RESULTS_DIR,
                        help="Root results directory (sub-dirs created automatically).")
    parser.add_argument("--top-maps",     type=int,   default=config.TUNING_TOP_MAPS,
                        help="Number of top-scoring configs to fully visualise.")
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                        help="Resume an interrupted hyperparameter sweep.")
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="Override checkpoint file path.")
    parser.add_argument("--memory-cap-mb",   type=float, default=6144.0)
    parser.add_argument("--no-cuda",     action="store_true",
                        help="Disable GPU acceleration (fall back to CPU).")
    parser.add_argument("--skip-sweep",  action="store_true",
                        help="Skip hyperparameter sweep; use existing checkpoint CSV.")
    parser.add_argument("--best-k",     type=int,   default=config.CLIQUE_GRID_DIVISIONS_K,
                        help="K to use for final best-config run (default: paper K=30).")
    parser.add_argument("--best-r",     type=float, default=config.CLIQUE_DENSITY_THRESHOLD_R,
                        help="r to use for final best-config run.")
    parser.add_argument("--best-nbr",   type=str,   default=config.CLIQUE_NEIGHBOR_MODE,
                        help="Neighbor mode for final best-config run.")
    args = parser.parse_args()

    use_cuda = not args.no_cuda
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Directory layout ────────────────────────────────────────────────────────
    tuning_dir  = os.path.join(args.results_dir, "tuning")
    maps_dir    = os.path.join(args.results_dir, "maps")
    runs_dir    = os.path.join(args.results_dir, "runs", ts)
    for d in (tuning_dir, maps_dir, runs_dir):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(runs_dir, "run_plsn_full.log")
    setup_logging(log_path)
    logger = logging.getLogger("PLSN-Full")

    logger.info("=== PLSN Full Pipeline Run ===")
    logger.info("  Timestamp:     %s", ts)
    logger.info("  Data:          %s", args.data_path)
    logger.info("  Results root:  %s", args.results_dir)
    logger.info("  CUDA:          %s", use_cuda)

    # ── 1. Load & preprocess ────────────────────────────────────────────────────
    logger.info("[1/5] Loading AIS data …")
    df = AISDataLoader(args.data_path).load_data()
    stationary_df = AISPreprocessor(
        sog_threshold=config.SOG_THRESHOLD,
        nav_status_filter=config.NAV_STATUS_FILTER,
    ).filter_anchor_mooring(df)
    logger.info("  Stationary points: %d", len(stationary_df))

    lat_col = "LAT" if "LAT" in df.columns else "Latitude"
    lon_col = "LON" if "LON" in df.columns else "Longitude"
    map_full_df = df[[lat_col, lon_col]].dropna()
    if len(map_full_df) > 400_000:
        map_full_df = map_full_df.sample(n=400_000, random_state=42)

    # ── 2. Hyperparameter sweep ─────────────────────────────────────────────────
    checkpoint_path = (
        args.checkpoint_path.strip()
        or os.path.join(tuning_dir, "sweep_checkpoint.csv")
    )

    if not args.skip_sweep:
        logger.info("[2/5] Running hyperparameter sweep …")
        tuning_cfg = TuningConfig(
            k_values=config.TUNING_K_VALUES,
            r_values=config.TUNING_R_VALUES,
            neighbor_modes=config.TUNING_NEIGHBOR_MODES,
            min_dense_points_values=config.TUNING_MIN_DENSE_POINTS,
            min_port_points=config.TUNING_MIN_PORT_POINTS,
            expected_ports_min=config.TUNING_EXPECTED_PORTS_MIN,
            expected_ports_max=config.TUNING_EXPECTED_PORTS_MAX,
        )
        tuner = CLIQUEHyperparameterTuner(tuning_dir, require_cuda=use_cuda, show_progress=True)
        results_df = tuner.run_sweep(
            stationary_df, tuning_cfg,
            checkpoint_path=checkpoint_path,
            resume=args.resume_from_checkpoint,
            memory_cap_mb=args.memory_cap_mb if args.memory_cap_mb > 0 else None,
        )
        csv_path, summary_path = tuner.export_results(results_df, prefix=f"plsn_tuning_{ts}")
        logger.info("  Sweep → %s", csv_path)
    else:
        logger.info("[2/5] Skipping sweep — loading checkpoint: %s", checkpoint_path)
        if not os.path.exists(checkpoint_path):
            logger.error("Checkpoint not found: %s", checkpoint_path)
            sys.exit(1)
        results_df = pd.read_csv(checkpoint_path)
        results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
        logger.info("  Loaded %d trials from checkpoint.", len(results_df))

    if results_df.empty:
        logger.error("No tuning results. Aborting.")
        sys.exit(1)

    best_row = results_df.iloc[0]
    logger.info(
        "[2/5] Best: K=%d r=%.6f nbr=%s score=%.4f ports=%d",
        int(best_row["k"]), float(best_row["r"]),
        str(best_row["neighbor_mode"]), float(best_row["score"]),
        int(best_row["effective_ports"]),
    )

    # ── 3. Generate top-N detailed maps ────────────────────────────────────────
    logger.info("[3/5] Generating top-%d method maps …", args.top_maps)
    top_n = min(args.top_maps, len(results_df))
    selected = results_df.head(top_n).copy()
    top_maps_dir = os.path.join(maps_dir, f"top_{top_n}_{ts}")
    os.makedirs(top_maps_dir, exist_ok=True)

    method_rows: list[dict] = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Map generation", unit="map"):
        mname = method_id(row)
        method_dir = os.path.join(top_maps_dir, mname)
        logger.info("  Processing config: %s", mname)

        artefacts = run_pipeline_for_params(
            stationary_df=stationary_df,
            map_full_df=map_full_df,
            output_dir=method_dir,
            k=int(row["k"]),
            r=float(row["r"]),
            neighbor_mode=str(row["neighbor_mode"]),
            min_dense_points=None if pd.isna(row["min_dense_points"]) else int(row["min_dense_points"]),
            use_cuda=use_cuda,
            trial_score=float(row["score"]),
            logger=logger,
        )
        method_rows.append({
            "method_name":   mname,
            "k":             int(row["k"]),
            "r":             float(row["r"]),
            "neighbor_mode": str(row["neighbor_mode"]),
            "ports":         int(row["effective_ports"]),
            "edges":         int(row["edge_count"]),
            "transitions":   int(row["transition_count"]),
            "score":         float(row["score"]),
            "map_rel_path":  os.path.relpath(artefacts["map_html"], maps_dir),
        })

    # ── 4. Comparison dashboard ─────────────────────────────────────────────────
    logger.info("[4/5] Building comparison dashboard …")
    comparison_html = os.path.join(maps_dir, f"plsn_comparison_{ts}.html")
    build_comparison_html(
        comparison_html,
        results_df,
        method_rows,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    logger.info("  Dashboard: %s", comparison_html)

    # ── 5. Final best-config run (named artefacts) ──────────────────────────────
    logger.info("[5/5] Running final best-config pipeline (K=%d r=%.6f nbr=%s) …",
                args.best_k, args.best_r, args.best_nbr)
    final_artefacts = run_pipeline_for_params(
        stationary_df=stationary_df,
        map_full_df=map_full_df,
        output_dir=runs_dir,
        k=args.best_k,
        r=args.best_r,
        neighbor_mode=args.best_nbr,
        min_dense_points=config.CLIQUE_MIN_DENSE_POINTS,
        use_cuda=use_cuda,
        trial_score=None,
        logger=logger,
    )

    # ── Summary ─────────────────────────────────────────────────────────────────
    sep = "─" * 72
    print(f"\n{sep}")
    print("  PLSN Full Pipeline Complete")
    print(sep)
    print(f"  Comparison dashboard : {comparison_html}")
    print(f"  Best trial map       : {method_rows[0]['map_rel_path'] if method_rows else 'n/a'}")
    print(f"  Final run map        : {final_artefacts['map_html']}")
    print(f"  Final run nodes      : {final_artefacts['nodes_csv']}")
    print(f"  Final run edges      : {final_artefacts['edges_csv']}")
    print(f"  Ports / Edges / Trans: {final_artefacts['n_ports']} / "
          f"{final_artefacts['n_edges']} / {final_artefacts['n_transitions']}")
    print(sep)
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
