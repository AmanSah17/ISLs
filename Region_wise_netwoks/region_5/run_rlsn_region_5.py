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
    import config_region_5 as config
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
    logger = logging.getLogger("Region5_RLSN")

    logger.info("Starting RLSN Extraction for Region 5...")
    
    # 1. Load data
    loader = AISDataLoader(config.DATA_FILE_PATH)
    ais_df = loader.load_data()
    
    # 2. Load NLSN results (Best Gamma 0.02)
    nlsn_best_dir = os.path.join(config.NLSN_RESULTS_DIR, "gamma_0p02")
    nodes_df = pd.read_csv(os.path.join(nlsn_best_dir, "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(nlsn_best_dir, "edges.csv"))
    
    # 3. Generate RLSN
    generator = RLSNGenerator(output_dir=config.RLSN_OUTPUT_DIR)
    rlsn_routes, rlsn_boundaries = generator.extract_rlsn(ais_df, nodes_df, edges_df)
    
    logger.info("Region 5 RLSN Extraction completed. Generated %d routes.", len(rlsn_routes))

if __name__ == "__main__":
    main()
