import logging
import sys
import os
import argparse
import json
from datetime import datetime
import pandas as pd

# --- Path Configuration ---
REGION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add NLSN module path specifically
NLSN_MODULE_DIR = os.path.join(REPO_ROOT, "NLSN_Extraction")
if NLSN_MODULE_DIR not in sys.path:
    sys.path.insert(0, NLSN_MODULE_DIR)

# --- Imports ---
try:
    import config_region_3 as config
    from NLSN_Extraction.modules.nlsn_tuner import BestPLSNParams, NLSNGammaConfig, NLSNGammaTuner
    from NLSN_Extraction.run_nlsn_gamma_sweep import load_best_plsn_params
except ImportError as e:
    print(f"CRITICIAL IMPORT ERROR: {e}")
    sys.exit(1)

def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )

def main():
    parser = argparse.ArgumentParser(description="Region 3 NLSN gamma sweep.")
    parser.add_argument("--plsn-summary", type=str, default="", help="Path to best PLSN summary JSON.")
    parser.add_argument("--require-cuda", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(config.NLSN_RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(config.NLSN_RESULTS_DIR, "nlsn_gamma_sweep.log")
    setup_logging(log_path)
    logger = logging.getLogger("Region3-NLSN-GammaSweep")

    # 1. Load best PLSN params
    plsn_summary = args.plsn_summary
    if not plsn_summary:
        tuning_dir = config.TUNING_DIR
        summaries = [f for f in os.listdir(tuning_dir) if f.startswith("region3_plsn_tuning_") and f.endswith("_summary.json")]
        if summaries:
            summaries.sort()
            plsn_summary = os.path.join(tuning_dir, summaries[-1])
            logger.info("Auto-discovered latest PLSN summary: %s", plsn_summary)

    plsn_params = load_best_plsn_params(
        summary_json_path=plsn_summary,
        results_csv_path="",
        k_override=None,
        r_override=None,
        neighbor_override=None,
        min_dense_override=None
    )
    logger.info("Using PLSN params: k=%d, r=%.7f, neighbor=%s", plsn_params.k, plsn_params.r, plsn_params.neighbor_mode)

    # 2. Load and Preprocess for NLSN
    logger.info("Loading data from %s", config.DATA_FILE_PATH)
    df = pd.read_parquet(config.DATA_FILE_PATH)
    time_col = 'BaseDateTime'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(['MMSI', time_col])

    # 3. Configure Tuner
    cfg = NLSNGammaConfig(
        gamma_values=config.NLSN_GAMMA_VALUES,
        min_trajectory_points=config.NLSN_MIN_TRAJ_POINTS,
        max_time_gap_minutes=config.NLSN_MAX_TIME_GAP_MIN,
        alpha_shape=config.NLSN_ALPHA_SHAPE,
        w1=1.0, 
        w2=1.0, 
        expected_nodes_min=config.NLSN_EXPECTED_NODES_MIN,
        expected_nodes_max=config.NLSN_EXPECTED_NODES_MAX,
        map_sample_size=config.VISUALIZATION_SAMPLE_SIZE
    )

    tuner = NLSNGammaTuner(
        output_dir=config.NLSN_RESULTS_DIR,
        require_cuda=args.require_cuda,
        show_progress=True
    )

    # 4. Run Sweep
    logger.info("Starting NLSN Gamma Sweep for Region 3...")
    results_df = tuner.run_sweep(full_df=df, plsn_params=plsn_params, cfg=cfg)

    if results_df.empty:
        logger.warning("No NLSN results were produced.")
        return

    best = results_df.iloc[0].to_dict()
    logger.info("Run complete. Best gamma=%.6g score=%.4f nodes=%d edges=%d",
                float(best["gamma"]), float(best["score"]), int(best["nodes"]), int(best["edges"]))
    
    print(f"\n[OK] Region 3 NLSN Tuning completed.")

if __name__ == "__main__":
    main()
