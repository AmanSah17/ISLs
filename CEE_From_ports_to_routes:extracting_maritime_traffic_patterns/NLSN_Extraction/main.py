from __future__ import annotations

import logging
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from NLSN_Extraction import config
from NLSN_Extraction.run_nlsn_gamma_sweep import (
    _load_input_df,
    _estimate_parquet_rows,
    load_best_plsn_params,
    setup_logging,
)
from NLSN_Extraction.modules.nlsn_tuner import NLSNGammaConfig, NLSNGammaTuner


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(config.OUTPUT_DIR, "nlsn_pipeline.log")
    setup_logging(log_path)
    logger = logging.getLogger("NLSN-Main")

    logger.info("Starting standalone NLSN extraction pipeline from NLSN_Extraction package.")
    logger.info("Output dir: %s", config.OUTPUT_DIR)

    estimated_rows = _estimate_parquet_rows(config.FULL_DATA_PATH)
    if estimated_rows is not None:
        logger.info("Input parquet estimated rows: %d", estimated_rows)

    best_plsn = load_best_plsn_params(
        summary_json_path=config.PLSN_SUMMARY_JSON,
        results_csv_path=config.PLSN_RESULTS_CSV,
        k_override=None,
        r_override=None,
        neighbor_override=None,
        min_dense_override=None,
    )
    logger.info(
        "Resolved best PLSN credentials: k=%d r=%.7f neighbor_mode=%s min_dense_points=%s",
        best_plsn.k,
        best_plsn.r,
        best_plsn.neighbor_mode,
        str(best_plsn.min_dense_points),
    )

    full_df = _load_input_df(
        data_path=config.FULL_DATA_PATH,
        max_input_rows=config.MAX_INPUT_ROWS,
        sample_mmsi_count=config.SAMPLE_MMSI_COUNT,
        logger=logger,
    )

    gamma_cfg = NLSNGammaConfig(
        gamma_values=config.GAMMA_VALUES,
        min_trajectory_points=config.MIN_TRAJECTORY_POINTS,
        max_time_gap_minutes=config.MAX_TIME_GAP_MINUTES,
        alpha_shape=config.ALPHA_SHAPE,
        w1=config.WEIGHT_W1,
        w2=config.WEIGHT_W2,
        map_sample_size=config.MAP_SAMPLE_SIZE,
    )
    tuner = NLSNGammaTuner(
        output_dir=config.OUTPUT_DIR,
        require_cuda=config.REQUIRE_CUDA,
        show_progress=config.SHOW_PROGRESS,
    )
    results = tuner.run_sweep(full_df=full_df, plsn_params=best_plsn, cfg=gamma_cfg)

    if results.empty:
        logger.warning("No NLSN results produced.")
        return

    best = results.iloc[0]
    logger.info(
        "Best NLSN trial: gamma=%.6g score=%.4f nodes=%d edges=%d transitions=%d",
        float(best["gamma"]),
        float(best["score"]),
        int(best["nodes"]),
        int(best["edges"]),
        int(best["transitions"]),
    )
    logger.info("NLSN pipeline completed.")


if __name__ == "__main__":
    main()
