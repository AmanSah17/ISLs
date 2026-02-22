"""
Orchestrator for the Route-Level Shipping Network (RLSN) Pipeline.
Loads NLSN results and performs Gaussian traffic flow fitting on raw trajectories.
"""

import logging
import os
import time

import pandas as pd

from modules.rlsn_generator import RLSNGenerator

# Configuration
BASE_DIR = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
RESULTS_DIR = os.path.join(BASE_DIR, "PLSN_Extraction", "results")
AIS_SOURCE = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"
RLSN_OUTPUT_DIR = os.path.join(RESULTS_DIR, "rlsn", time.strftime("%Y%m%d_%H%M%S"))

# Hyperparameters
NUM_SLICES = 15      # Higher value = smoother routes
SEARCH_RADIUS = 0.05 # Degrees to look for points around each slice center
MAX_AIS_POINTS = 1_000_000 # Limit memory usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_nlsn_dir():
    nlsn_root = os.path.join(RESULTS_DIR, "nlsn")
    if not os.path.exists(nlsn_root): return None
    # Pick a representative NLSN trial
    dirs = [d for d in os.listdir(nlsn_root) if d.startswith("gamma_")]
    # Prefer gamma_0p001_w1_1p0_w2_1p0 if available
    preferred = [d for d in dirs if "gamma_0p001" in d]
    target = preferred[0] if preferred else (sorted(dirs)[0] if dirs else None)
    return os.path.join(nlsn_root, target) if target else None


def main():
    logger.info("Starting RLSN Pipeline...")
    
    # 1. Find NLSN Input
    nlsn_dir = find_latest_nlsn_dir()
    if not nlsn_dir:
        logger.error("No NLSN directory found. Run NLSN pipeline first.")
        return

    logger.info("Using NLSN source: %s", nlsn_dir)
    nodes_path = os.path.join(nlsn_dir, "nodes.csv")
    edges_path = os.path.join(nlsn_dir, "edges.csv")
    
    if not all(os.path.exists(p) for p in [nodes_path, edges_path]):
        logger.error("NLSN nodes or edges missing in %s", nlsn_dir)
        return

    # 2. Load AIS Data (Full trajectories)
    if not os.path.exists(AIS_SOURCE):
        logger.error("AIS source not found at %s", AIS_SOURCE)
        return

    logger.info("Loading AIS data from %s...", AIS_SOURCE)
    try:
        # We need LAT, LON for RLSN spatial filtering
        ais_df = pd.read_parquet(AIS_SOURCE, columns=["LAT", "LON"])
        if len(ais_df) > MAX_AIS_POINTS:
            ais_df = ais_df.sample(MAX_AIS_POINTS, random_state=42)
    except Exception as e:
        logger.error("Failed to load AIS data: %s", e)
        return

    # 3. Initialize Mapper
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    generator = RLSNGenerator(RLSN_OUTPUT_DIR, num_slices=NUM_SLICES, search_radius=SEARCH_RADIUS)
    
    # 4. Execute Extraction
    start_time = time.time()
    routes, boundaries = generator.extract_rlsn(ais_df, nodes_df, edges_df)
    elapsed = time.time() - start_time
    
    logger.info("RLSN Pipeline completed in %.2f seconds.", elapsed)
    print(f"\nSUCCESS: RLSN results saved to {RLSN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
