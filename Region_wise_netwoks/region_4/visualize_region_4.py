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
    import config_region_4 as config
    from integrated_visualization.integrated_visualizer import IntegratedVisualizer
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Generating Final Integrated Dashboard for Region 4...")
    output_html = os.path.join(REGION_DIR, "integrated_dashboard_region_4.html")
    visualizer = IntegratedVisualizer(output_html)
    
    if os.path.exists(config.DATA_FILE_PATH):
        ais_df = pd.read_parquet(config.DATA_FILE_PATH, columns=["LAT", "LON"])
        visualizer.set_heatmap_data(ais_df)
    
    maps_root = config.MAPS_DIR
    if os.path.exists(maps_root):
        folders = [f for f in os.listdir(maps_root) if f.startswith("top_") and os.path.isdir(os.path.join(maps_root, f))]
        if folders:
            folders.sort()
            latest_top = os.path.join(maps_root, folders[-1])
            for trial_dir in sorted(os.listdir(latest_top)):
                path = os.path.join(latest_top, trial_dir)
                if not os.path.isdir(path): continue
                n, e, g = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson"]]
                if all(os.path.exists(p) for p in [n,e,g]):
                    visualizer.add_plsn_layer(trial_dir, pd.read_csv(n), pd.read_csv(e), json.load(open(g)), None, batch="Macro Discovery")

    nlsn_root = config.NLSN_RESULTS_DIR
    if os.path.exists(nlsn_root):
        for trial_dir in sorted(os.listdir(nlsn_root)):
            path = os.path.join(nlsn_root, trial_dir)
            if not os.path.isdir(path) or not trial_dir.startswith("gamma_"): continue
            n, e, g = [os.path.join(path, f) for f in ["nodes.csv", "edges.csv", "boundaries.geojson"]]
            if all(os.path.exists(p) for p in [n,e,g]):
                visualizer.add_nlsn_layer(trial_dir, pd.read_csv(n), pd.read_csv(e), json.load(open(g)), None, batch="Meso Workstation")

    rlsn_dir = config.RLSN_OUTPUT_DIR
    routes_path = os.path.join(rlsn_dir, "rlsn_routes.geojson")
    bounds_path = os.path.join(rlsn_dir, "rlsn_boundaries.geojson")
    if all(os.path.exists(p) for p in [routes_path, bounds_path]):
        visualizer.add_rlsn_layer("Region 4 RLSN", json.load(open(routes_path)), json.load(open(bounds_path)), batch="Route Corridors")
    
    visualizer.generate_dashboard()
    logger.info("SUCCESS: Region 4 Integrated Dashboard generated at %s", output_html)

if __name__ == "__main__":
    main()
