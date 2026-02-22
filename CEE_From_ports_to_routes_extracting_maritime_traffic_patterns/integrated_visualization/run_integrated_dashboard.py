"""
Orchestrator to generate the Integrated Multiscale Research Portal.
Discovers multiple hyperparameter tuning results and integrates technical analysis.
"""

import json
import logging
import os

import pandas as pd

from integrated_visualizer import IntegratedVisualizer

# Config
BASE_DIR = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
RESULTS_DIR = os.path.join(BASE_DIR, "PLSN_Extraction", "results")
AIS_SOURCE_PARQUET = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"
RLSN_RESULTS_DIR = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\RLSN_Extraction\results"

METHODOLOGY_PATH = os.path.join(BASE_DIR, "METHODOLOGY.md")
OUTPUT_PATH = os.path.join(BASE_DIR, "integrated_visualization", "integrated_dashboard.html")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_plsn_top_dir():
    return os.path.join(RESULTS_DIR, "maps", "top_15_20260220_215918")

def find_nlsn_top_dir():
    return os.path.join(RESULTS_DIR, "nlsn")


def main():
    visualizer = IntegratedVisualizer(OUTPUT_PATH)

    # 1. Load Methodology
    if os.path.exists(METHODOLOGY_PATH):
        logger.info("Loading methodology from %s", METHODOLOGY_PATH)
        with open(METHODOLOGY_PATH, "r", encoding="utf-8") as f:
            visualizer.set_methodology(f.read())

    # 2. Load Heatmap
    if os.path.exists(AIS_SOURCE_PARQUET):
        logger.info("Loading AIS heatmap data from PARQUET: %s", AIS_SOURCE_PARQUET)
        try:
            ais_df = pd.read_parquet(AIS_SOURCE_PARQUET, columns=["LAT", "LON"])
            visualizer.set_heatmap_data(ais_df)
        except Exception as e:
            logger.error("Failed to load Parquet AIS: %s", e)
    elif os.path.exists(AIS_SOURCE_CSV):
        logger.info("Loading AIS heatmap data from CSV: %s", AIS_SOURCE_CSV)
        try:
            ais_df = pd.read_csv(AIS_SOURCE_CSV, usecols=["LAT", "LON"], nrows=500_000)
            visualizer.set_heatmap_data(ais_df)
        except Exception as e:
            logger.error("Failed to load CSV AIS: %s", e)
    else:
        logger.warning("AIS source not found. Heatmap will be limited.")

    # 3. Add PLSN Layers (Batch A: Port Discovery)
    plsn_dir = find_plsn_top_dir()
    fallback_points = None
    if os.path.exists(AIS_SOURCE_PARQUET):
        try:
            fallback_points = pd.read_parquet(AIS_SOURCE_PARQUET, columns=["LAT", "LON", "MMSI", "BaseDateTime"]).sample(n=10000, random_state=42).rename(columns={"MMSI": "TRAJECTORY_ID", "BaseDateTime": "TIMESTAMP"})
        except Exception as e:
            logger.error("Failed to sample fallback Parquet: %s", e)

    if os.path.exists(plsn_dir):
        logger.info("  Adding PLSN Layers from %s", plsn_dir)
        for config_dir in sorted(os.listdir(plsn_dir)):
            path = os.path.join(plsn_dir, config_dir)
            if not os.path.isdir(path): continue
            nodes_path, edges_path, geo_path, points_path = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson", "feature_points.csv"]]
            if all(os.path.exists(p) for p in [nodes_path, edges_path, geo_path]):
                trial_points = pd.read_csv(points_path) if os.path.exists(points_path) else fallback_points
                visualizer.add_plsn_layer(config_dir, pd.read_csv(nodes_path), pd.read_csv(edges_path), json.load(open(geo_path)), trial_points, batch="Macro Discovery")

    # 4. Add NLSN Layers (Batch B: Meso Waypoints)
    nlsn_dir = find_nlsn_top_dir()
    if os.path.exists(nlsn_dir):
        logger.info("  Adding NLSN Layers from %s", nlsn_dir)
        for config_dir in sorted(os.listdir(nlsn_dir)):
            path = os.path.join(nlsn_dir, config_dir)
            if not os.path.isdir(path) or not config_dir.startswith("gamma_"): continue
            nodes_path, edges_path, geo_path, points_path = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson", "feature_points.csv"]]
            if all(os.path.exists(p) for p in [nodes_path, edges_path, geo_path]):
                points = pd.read_csv(points_path) if os.path.exists(points_path) else None
                visualizer.add_nlsn_layer(config_dir, pd.read_csv(nodes_path), pd.read_csv(edges_path), json.load(open(geo_path)), points, batch="Meso Workstation")

    # 5. Add Standalone RLSN Layers (Batch C: Route Extraction)
    if os.path.exists(RLSN_RESULTS_DIR):
        logger.info("  Adding RLSN Layers from %s", RLSN_RESULTS_DIR)
        for run_id in sorted(os.listdir(RLSN_RESULTS_DIR)):
            path = os.path.join(RLSN_RESULTS_DIR, run_id)
            if not os.path.isdir(path): continue
            routes_path = os.path.join(path, "rlsn_routes.geojson")
            bounds_path = os.path.join(path, "rlsn_boundaries.geojson")
            if all(os.path.exists(p) for p in [routes_path, bounds_path]):
                visualizer.add_rlsn_layer(run_id, json.load(open(routes_path)), json.load(open(bounds_path)), batch="Route Corridors")

    # 6. Generate Portal
    visualizer.generate_dashboard()
    print(f"\nSUCCESS: Maritime Research Portal generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
