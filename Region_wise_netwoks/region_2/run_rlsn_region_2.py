import logging
import sys
import os
import pandas as pd
import numpy as np

# --- Path Configuration ---
REGION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Imports ---
try:
    import config_region_2 as config
    from PLSN_Extraction.modules.data_loader import AISDataLoader
    from PLSN_Extraction.modules.preprocessor import AISPreprocessor
    from PLSN_Extraction.modules.clustering import CLIQUEClusterer
    from PLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
    from PLSN_Extraction.modules.rlsn_generator import RLSNGenerator
    from PLSN_Extraction.modules.visualizer import PLSNVisualizer as Visualizer
except ImportError as e:
    print(f"CRITICIAL IMPORT ERROR: {e}")
    sys.exit(1)

def main():
    if not os.path.exists(config.RLSN_OUTPUT_DIR):
        os.makedirs(config.RLSN_OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Region2_RLSN")

    logger.info("Starting RLSN Extraction for Region 2...")
    
    # 1. Load data
    loader = AISDataLoader(config.DATA_FILE_PATH)
    ais_df = loader.load_data()
    logger.info("Loaded AIS data: %d rows", len(ais_df))
    
    # 2. Load NLSN results (Best Gamma 0.05)
    nlsn_best_dir = os.path.join(config.NLSN_RESULTS_DIR, "gamma_0p05")
    nodes_df = pd.read_csv(os.path.join(nlsn_best_dir, "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(nlsn_best_dir, "edges.csv"))
    logger.info("Loaded NLSN: %d nodes, %d edges", len(nodes_df), len(edges_df))
    
    # 3. Generate RLSN
    generator = RLSNGenerator(output_dir=config.RLSN_OUTPUT_DIR)
    rlsn_routes, rlsn_boundaries = generator.extract_rlsn(ais_df, nodes_df, edges_df)
    
    logger.info("Region 2 RLSN Extraction completed. Extracted routes: %d, Boundaries: %d", len(rlsn_routes), len(rlsn_boundaries))

if __name__ == "__main__":
    main()
