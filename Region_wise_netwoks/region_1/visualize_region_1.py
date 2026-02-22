import sys
import os
import json
import pandas as pd
import logging

# --- Path Configuration ---
REGION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Imports ---
try:
    import config_region_1 as config
    from integrated_visualization.integrated_visualizer import IntegratedVisualizer
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Generating Final Integrated Dashboard for Region 1...")
    
    output_html = os.path.join(REGION_DIR, "integrated_dashboard_region_1.html")
    visualizer = IntegratedVisualizer(output_html)
    
    # 1. Load and sample AIS for heatmap
    if os.path.exists(config.DATA_FILE_PATH):
        logger.info("Loading AIS heatmap data from PARQUET: %s", config.DATA_FILE_PATH)
        ais_df = pd.read_parquet(config.DATA_FILE_PATH, columns=["LAT", "LON"])
        visualizer.set_heatmap_data(ais_df)
    
    # 2. Add PLSN Layers (from tuning search)
    # We look into the maps folder created during tuning
    maps_root = config.MAPS_DIR
    if os.path.exists(maps_root):
        # find the latest top_N folder
        folders = [f for f in os.listdir(maps_root) if f.startswith("top_") and os.path.isdir(os.path.join(maps_root, f))]
        if folders:
            folders.sort()
            latest_top = os.path.join(maps_root, folders[-1])
            logger.info("Discovering PLSN Layers from %s", latest_top)
            for trial_dir in sorted(os.listdir(latest_top)):
                path = os.path.join(latest_top, trial_dir)
                if not os.path.isdir(path): continue
                nodes_path, edges_path, geo_path = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson"]]
                if all(os.path.exists(p) for p in [nodes_path, edges_path, geo_path]):
                    visualizer.add_plsn_layer(trial_dir, pd.read_csv(nodes_path), pd.read_csv(edges_path), json.load(open(geo_path)), None, batch="Macro Discovery")

    # 3. Add NLSN Layers (from gamma sweep)
    nlsn_root = config.NLSN_RESULTS_DIR
    if os.path.exists(nlsn_root):
        logger.info("Discovering NLSN Layers from %s", nlsn_root)
        for trial_dir in sorted(os.listdir(nlsn_root)):
            path = os.path.join(nlsn_root, trial_dir)
            if not os.path.isdir(path) or not trial_dir.startswith("gamma_"): continue
            nodes_path, edges_path, geo_path = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson"]]
            if all(os.path.exists(p) for p in [nodes_path, edges_path, geo_path]):
                visualizer.add_nlsn_layer(trial_dir, pd.read_csv(nodes_path), pd.read_csv(edges_path), json.load(open(geo_path)), None, batch="Meso Workstation")

    # 4. Add RLSN Layer
    rlsn_dir = config.RLSN_OUTPUT_DIR
    routes_path = os.path.join(rlsn_dir, "rlsn_routes.geojson")
    bounds_path = os.path.join(rlsn_dir, "rlsn_boundaries.geojson")
    if all(os.path.exists(p) for p in [routes_path, bounds_path]):
        logger.info("Adding RLSN Layer from %s", rlsn_dir)
        visualizer.add_rlsn_layer("Region 1 RLSN", json.load(open(routes_path)), json.load(open(bounds_path)), batch="Route Corridors")
    
    # 5. Generate Portal
    visualizer.generate_dashboard()
    logger.info("SUCCESS: Region 1 Integrated Dashboard generated at %s", output_html)

if __name__ == "__main__":
    main()
