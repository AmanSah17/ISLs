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
AIS_SOURCE = r"f:\PyTorch_GPU\ISLs\ais_data\ais_data_2020_01_sorted.csv"
if not os.path.exists(AIS_SOURCE):
    # Fallback to relative if root search failed (though root should work if file exists)
    AIS_SOURCE = os.path.join(os.path.dirname(BASE_DIR), "ais_data", "ais_data_2020_01_sorted.csv")

METHODOLOGY_PATH = os.path.join(BASE_DIR, "METHODOLOGY.md")
OUTPUT_PATH = os.path.join(BASE_DIR, "integrated_visualization", "Research_Portal.html")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_plsn_top_dir():
    maps_dir = os.path.join(RESULTS_DIR, "maps")
    if not os.path.exists(maps_dir): return None
    dirs = [d for d in os.listdir(maps_dir) if d.startswith("top_15_")]
    return os.path.join(maps_dir, sorted(dirs)[-1]) if dirs else None


def main():
    visualizer = IntegratedVisualizer(OUTPUT_PATH)

    # 1. Load Methodology
    if os.path.exists(METHODOLOGY_PATH):
        logger.info("Loading methodology from %s", METHODOLOGY_PATH)
        with open(METHODOLOGY_PATH, "r", encoding="utf-8") as f:
            visualizer.set_methodology(f.read())

    # 2. Load Heatmap
    if os.path.exists(AIS_SOURCE):
        logger.info("Loading AIS heatmap data from %s", AIS_SOURCE)
        try:
            ais_df = pd.read_csv(AIS_SOURCE, usecols=["LAT", "LON"], nrows=500_000)
            visualizer.set_heatmap_data(ais_df)
        except Exception as e:
            logger.error("Failed to load AIS: %s", e)
    else:
        logger.warning("AIS source not found at %s. Heatmap will be limited.", AIS_SOURCE)

    # 3. Add PLSN Layers
    plsn_top_dir = find_latest_plsn_top_dir()
    if plsn_top_dir:
        for config_dir in sorted(os.listdir(plsn_top_dir)):
            path = os.path.join(plsn_top_dir, config_dir)
            if not os.path.isdir(path): continue
            nodes_path, edges_path, geo_path = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson"]]
            if all(os.path.exists(p) for p in [nodes_path, edges_path, geo_path]):
                logger.info("  Adding PLSN Layer: %s", config_dir)
                visualizer.add_plsn_layer(config_dir, pd.read_csv(nodes_path), pd.read_csv(edges_path), json.load(open(geo_path)))

    # 4. Add NLSN Layers
    nlsn_root = os.path.join(RESULTS_DIR, "nlsn")
    if os.path.exists(nlsn_root):
        for config_dir in sorted(os.listdir(nlsn_root)):
            path = os.path.join(nlsn_root, config_dir)
            if not os.path.isdir(path) or not config_dir.startswith("gamma_"): continue
            nodes_path, edges_path, geo_path, points_path = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson", "feature_points.csv"]]
            if all(os.path.exists(p) for p in [nodes_path, edges_path, geo_path]):
                logger.info("  Adding NLSN Layer: %s", config_dir)
                points = pd.read_csv(points_path) if os.path.exists(points_path) else None
                visualizer.add_nlsn_layer(config_dir, pd.read_csv(nodes_path), pd.read_csv(edges_path), json.load(open(geo_path)), points)

    # 5. Generate Portal
    visualizer.generate_dashboard()
    print(f"\nSUCCESS: Maritime Research Portal generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
