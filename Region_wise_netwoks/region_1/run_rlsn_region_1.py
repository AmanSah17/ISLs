import logging
import sys
import os

# --- Path Configuration ---
REGION_1_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add RLSN module path specifically since it might be separate
RLSN_MODULE_DIR = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\RLSN_Extraction"
if RLSN_MODULE_DIR not in sys.path:
    sys.path.insert(0, RLSN_MODULE_DIR)

# --- Imports ---
try:
    import pandas as pd
    import json
    import config_region_1 as config
    from PLSN_Extraction.modules.data_loader import AISDataLoader
    from modules.rlsn_generator import RLSNGenerator
    from modules.visualizer import RLSNVisualizer
except ImportError as e:
    print(f"CRITICIAL IMPORT ERROR: {e}")
    sys.exit(1)

def main():
    # Setup output directory
    if not os.path.exists(config.RLSN_OUTPUT_DIR):
        os.makedirs(config.RLSN_OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Region1_RLSN")
    
    logger.info("Starting RLSN Extraction for Region 1...")
    
    # 1. Load AIS data for RLSN
    loader = AISDataLoader(config.DATA_FILE_PATH)
    ais_df = loader.load_data()
    
    # paths to PLSN results
    plsn_edges = config.EDGES_CSV
    plsn_nodes = config.NODES_CSV
    
    if not os.path.exists(plsn_nodes) or not os.path.exists(plsn_edges):
        logger.error(f"Missing PLSN results: {plsn_nodes} or {plsn_edges}")
        return

    nodes_df = pd.read_csv(plsn_nodes)
    edges_df = pd.read_csv(plsn_edges)
    
    generator = RLSNGenerator(output_dir=config.RLSN_OUTPUT_DIR)
    rlsn_routes, rlsn_boundaries = generator.extract_rlsn(ais_df, nodes_df, edges_df)
    
    # RLSN visualizer - uses generate_map
    map_file = os.path.join(config.RLSN_OUTPUT_DIR, "rlsn_map.html")
    visualizer = RLSNVisualizer(output_file=map_file)
    
    # Load geojson files for visualization as expected by generate_map
    routes_path = os.path.join(config.RLSN_OUTPUT_DIR, "rlsn_routes.geojson")
    bounds_path = os.path.join(config.RLSN_OUTPUT_DIR, "rlsn_boundaries.geojson")
    
    with open(routes_path, 'r') as f: routes_gj = json.load(f)
    with open(bounds_path, 'r') as f: bounds_gj = json.load(f)
    
    visualizer.generate_map(routes_gj, bounds_gj)
    
    logger.info("Region 1 RLSN Extraction completed.")

if __name__ == "__main__":
    main()
