"""
Batch RLSN Runner — Iterates through all NLSN trials and extracts route-level patterns.
Saves RLSN metadata directly into the corresponding NLSN trial folders.
"""

import logging
import os
import time
import pandas as pd
import json

from modules.rlsn_generator import RLSNGenerator

# Configuration
BASE_DIR = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
RESULTS_DIR = os.path.join(BASE_DIR, "PLSN_Extraction", "results")
AIS_SOURCE = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"

# RLSN Hyperparameters
NUM_SLICES = 15
SEARCH_RADIUS = 0.05
MAX_AIS_POINTS = 1_000_000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_batch_rlsn():
    nlsn_root = os.path.join(RESULTS_DIR, "nlsn")
    if not os.path.exists(nlsn_root):
        logger.error("NLSN root directory not found.")
        return

    # 1. Load AIS Data Once
    logger.info("Loading AIS data for batch processing...")
    try:
        ais_df = pd.read_parquet(AIS_SOURCE, columns=["LAT", "LON"])
        if len(ais_df) > MAX_AIS_POINTS:
            ais_df = ais_df.sample(MAX_AIS_POINTS, random_state=42)
    except Exception as e:
        logger.error("Failed to load AIS data: %s", e)
        return

    # 2. Iterate through NLSN Trials
    trials = [d for d in os.listdir(nlsn_root) if os.path.isdir(os.path.join(nlsn_root, d)) and d.startswith("gamma_")]
    logger.info("Found %d NLSN trials for RLSN extraction.", len(trials))

    for trial in trials:
        trial_path = os.path.join(nlsn_root, trial)
        nodes_path = os.path.join(trial_path, "nodes.csv")
        edges_path = os.path.join(trial_path, "edges.csv")

        if not all(os.path.exists(p) for p in [nodes_path, edges_path]):
            logger.warning("  Skipping %s: Essential NLSN CSVs missing.", trial)
            continue

        logger.info("  Processing RLSN for scale: %s", trial)
        
        # Initialize generator with the trial folder as output
        generator = RLSNGenerator(trial_path, num_slices=NUM_SLICES, search_radius=SEARCH_RADIUS)
        
        try:
            nodes_df = pd.read_csv(nodes_path)
            edges_df = pd.read_csv(edges_path)
            
            start_time = time.time()
            routes, boundaries = generator.extract_rlsn(ais_df, nodes_df, edges_df)
            elapsed = time.time() - start_time
            
            logger.info("  -> Completed %s in %.2f seconds.", trial, elapsed)
        except Exception as e:
            logger.error("  -> Failed to process %s: %s", trial, e)

    logger.info("Batch RLSN extraction complete.")

if __name__ == "__main__":
    run_batch_rlsn()
