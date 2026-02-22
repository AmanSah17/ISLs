import logging
import sys
import os
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# --- Path Configuration ---
REGION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Imports ---
try:
    import config_region_3 as config
    from PLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
    from PLSN_Extraction.modules.clustering import CLIQUEClusterer
    from PLSN_Extraction.modules.data_loader import AISDataLoader
    from PLSN_Extraction.modules.hyperparameter_tuner import CLIQUEHyperparameterTuner, TuningConfig
    from PLSN_Extraction.modules.network_generator import PLSNGenerator
    from PLSN_Extraction.modules.preprocessor import AISPreprocessor
    from PLSN_Extraction.modules.visualizer import PLSNVisualizer
    from PLSN_Extraction.run_hyperparam_comparison import build_comparison_html, method_id
except ImportError as e:
    print(f"CRITICIAL IMPORT ERROR: {e}")
    sys.exit(1)

def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )

def main():
    parser = argparse.ArgumentParser(description="Region 3 PLSN hyperparameter sweep.")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU mode.")
    args = parser.parse_args()

    use_cuda = not args.no_cuda
    os.makedirs(config.MAPS_DIR, exist_ok=True)
    os.makedirs(config.TUNING_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.TUNING_DIR, f"plsn_tuning_{ts}.log")
    setup_logging(log_path)
    logger = logging.getLogger("Region3-PLSN-Tuning")

    # 1. Load & preprocess
    logger.info("Loading data from %s", config.DATA_FILE_PATH)
    df = AISDataLoader(config.DATA_FILE_PATH).load_data()
    stationary_df = AISPreprocessor(
        sog_threshold=config.SOG_THRESHOLD,
        nav_status_filter=config.NAV_STATUS_FILTER,
    ).filter_anchor_mooring(df)
    logger.info("Stationary candidate rows: %d", len(stationary_df))

    lat_col = "LAT" if "LAT" in df.columns else "Latitude"
    lon_col = "LON" if "LON" in df.columns else "Longitude"
    map_full_df = df[[lat_col, lon_col]].dropna()
    if len(map_full_df) > config.VISUALIZATION_SAMPLE_SIZE:
        map_full_df = map_full_df.sample(n=config.VISUALIZATION_SAMPLE_SIZE, random_state=42)

    # 2. Hyperparameter sweep
    tuning_cfg = TuningConfig(
        k_values=config.TUNING_K_VALUES,
        r_values=config.TUNING_R_VALUES,
        neighbor_modes=config.TUNING_NEIGHBOR_MODES,
        min_dense_points_values=config.TUNING_MIN_DENSE_POINTS,
        min_port_points=config.TUNING_MIN_PORT_POINTS,
        expected_ports_min=config.TUNING_EXPECTED_PORTS_MIN,
        expected_ports_max=config.TUNING_EXPECTED_PORTS_MAX,
    )
    tuner = CLIQUEHyperparameterTuner(
        config.TUNING_DIR, require_cuda=use_cuda, show_progress=True
    )
    checkpoint_path = os.path.join(config.TUNING_DIR, "plsn_tuning_checkpoint.csv")
    results_df = tuner.run_sweep(
        stationary_df,
        tuning_cfg,
        checkpoint_path=checkpoint_path,
        resume=False
    )
    csv_path, summary_path = tuner.export_results(results_df, prefix=f"region3_plsn_tuning_{ts}")

    if results_df.empty:
        logger.error("No tuning results. Exiting.")
        return

    # 3. Build detailed maps for top-N
    top_n = min(config.TUNING_TOP_MAPS, len(results_df))
    selected = results_df.head(top_n).copy()
    top_maps_dir = os.path.join(config.MAPS_DIR, f"top_{top_n}_{ts}")
    os.makedirs(top_maps_dir, exist_ok=True)

    method_rows: list[dict] = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Map generation", unit="map"):
        mname = method_id(row)
        method_dir = os.path.join(top_maps_dir, mname)
        os.makedirs(method_dir, exist_ok=True)

        clusterer = CLIQUEClusterer(
            k=int(row["k"]),
            density_threshold_r=float(row["r"]),
            min_dense_points=None if pd.isna(row["min_dense_points"]) else int(row["min_dense_points"]),
            neighbor_mode=str(row["neighbor_mode"]),
            require_cuda=use_cuda,
        )
        clustered_df = clusterer.fit_predict(stationary_df)

        boundaries = BoundaryExtractor(alpha=config.ALPHA_SHAPE_PARAMETER).extract_boundaries(clustered_df)
        generator  = PLSNGenerator(output_dir=method_dir)
        nodes_df   = generator.export_nodes_and_boundaries(boundaries, clustered_df=clustered_df)
        if nodes_df is None or nodes_df.empty:
            nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "area_deg2", "stationary_points"])
        valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
        edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)

        params_meta = {
            "k": int(row["k"]),
            "r": float(row["r"]),
            "neighbor_mode": str(row["neighbor_mode"]),
            "min_dense_points": None if pd.isna(row["min_dense_points"]) else int(row["min_dense_points"]),
        }
        map_path  = os.path.join(method_dir, "plsn_map.html")
        visualizer = PLSNVisualizer(
            output_file=map_path,
            sample_size=config.VISUALIZATION_SAMPLE_SIZE,
            edge_min_width=config.EDGE_MIN_WIDTH,
            edge_max_width=config.EDGE_MAX_WIDTH,
            boundary_weight=config.BOUNDARY_WEIGHT,
            node_min_radius=config.NODE_MIN_RADIUS,
            node_max_radius=config.NODE_MAX_RADIUS,
        )
        visualizer.generate_plsn_dashboard(
            map_full_df, clustered_df, boundaries, nodes_df, edges_df,
            params_meta=params_meta, trial_score=float(row["score"]),
        )

        method_rows.append({
            "method_name": mname,
            "k": int(row["k"]),
            "r": float(row["r"]),
            "neighbor_mode": str(row["neighbor_mode"]),
            "ports": int(row["effective_ports"]),
            "edges": int(row["edge_count"]),
            "transitions": int(row["transition_count"]),
            "score": float(row["score"]),
            "map_rel_path": os.path.relpath(map_path, config.MAPS_DIR),
        })

    comparison_html = os.path.join(config.MAPS_DIR, f"plsn_comparison_{ts}.html")
    build_comparison_html(
        comparison_html,
        results_df,
        method_rows,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    print(f"\n[OK] Region 3 PLSN Tuning completed.")

if __name__ == "__main__":
    main()
