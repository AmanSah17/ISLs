"""
run_nlsn_full.py — End-to-end NLSN pipeline orchestrator.

Workflow
────────
1. Load full AIS data (all points, not just stationary)
2. Read best PLSN params from tuning CSV (or use --best-k/r/nbr overrides)
3. Run extensive gamma × w1 × w2 sweep via NLSNGammaTuner
4. Generate enhanced comparison dashboard → results/nlsn/nlsn_comparison_<ts>.html
5. Summary printed to console

Usage
─────
  python run_nlsn_full.py                          # uses config.py defaults
  python run_nlsn_full.py --no-cuda               # CPU-only
  python run_nlsn_full.py --best-k 1400 --best-r 0.000005 --best-nbr 4
  python run_nlsn_full.py --gamma-only            # only 12 gammas, w1=w2=1 (fast)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from PLSN_Extraction import config
from PLSN_Extraction.modules.data_loader import AISDataLoader
from PLSN_Extraction.modules.nlsn_tuner import (
    BestPLSNParams,
    NLSNGammaConfig,
    NLSNGammaTuner,
)


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    for lib in ("numba", "numba.cuda", "numba.cuda.cudadrv.driver", "pyogrio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger("NLSN-Full")


# ── Best PLSN param helper ─────────────────────────────────────────────────────
def _load_best_plsn_params(tuning_dir: str) -> BestPLSNParams | None:
    """
    Scan results/tuning/ for the latest plsn_tuning_*_results.csv,
    load it, and return the top-scoring row as BestPLSNParams.
    """
    import glob
    pattern = os.path.join(tuning_dir, "plsn_tuning_*_results.csv")
    candidates = sorted(glob.glob(pattern), reverse=True)
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    if df.empty:
        return None
    best = df.sort_values("score", ascending=False).iloc[0]
    return BestPLSNParams(
        k             = int(best["k"]),
        r             = float(best["r"]),
        neighbor_mode = str(best["neighbor_mode"]),
        min_dense_points = None if pd.isna(best.get("min_dense_points", float("nan"))) else int(best["min_dense_points"]),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end NLSN pipeline: gamma+w1+w2 sweep -> maps -> comparison dashboard."
    )
    parser.add_argument("--data-path",  type=str,   default=config.DATA_FILE_PATH)
    parser.add_argument("--nlsn-dir",   type=str,   default=config.NLSN_DIR)
    parser.add_argument("--tuning-dir", type=str,   default=config.TUNING_DIR,
                        help="Dir with PLSN tuning CSVs for auto-loading best PLSN params.")
    parser.add_argument("--best-k",    type=int,   default=None, help="Override PLSN K.")
    parser.add_argument("--best-r",    type=float, default=None, help="Override PLSN r.")
    parser.add_argument("--best-nbr",  type=str,   default=None, help="Override PLSN neighbor mode.")
    parser.add_argument("--no-cuda",   action="store_true")
    parser.add_argument("--gamma-only", action="store_true",
                        help="Sweep only gamma (w1=w2=1.0). Runs 12 trials instead of 192.")
    parser.add_argument("--top-maps",  type=int,   default=config.NLSN_TOP_MAPS)
    args = parser.parse_args()

    use_cuda = not args.no_cuda
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.nlsn_dir, exist_ok=True)

    log_path = os.path.join(args.nlsn_dir, f"run_nlsn_{ts}.log")
    logger   = setup_logging(log_path)

    logger.info("=== NLSN Full Pipeline Run ===")
    logger.info("  Timestamp : %s", ts)
    logger.info("  Data      : %s", args.data_path)
    logger.info("  NLSN dir  : %s", args.nlsn_dir)
    logger.info("  CUDA      : %s", use_cuda)

    # ── 1. Resolve best PLSN params ─────────────────────────────────────────────
    plsn_params: BestPLSNParams | None = None
    if args.best_k and args.best_r and args.best_nbr:
        plsn_params = BestPLSNParams(k=args.best_k, r=args.best_r, neighbor_mode=args.best_nbr)
        logger.info("[1/4] Using CLI PLSN params: K=%d r=%.6f nbr=%s", args.best_k, args.best_r, args.best_nbr)
    else:
        plsn_params = _load_best_plsn_params(args.tuning_dir)
        if plsn_params:
            logger.info("[1/4] Loaded best PLSN from tuning CSV: K=%d r=%.6f nbr=%s",
                        plsn_params.k, plsn_params.r, plsn_params.neighbor_mode)
        else:
            # fallback to paper defaults
            plsn_params = BestPLSNParams(
                k=config.CLIQUE_GRID_DIVISIONS_K,
                r=config.CLIQUE_DENSITY_THRESHOLD_R,
                neighbor_mode=config.CLIQUE_NEIGHBOR_MODE,
            )
            logger.warning("[1/4] No PLSN tuning CSV found; using paper defaults K=%d r=%.6f",
                           plsn_params.k, plsn_params.r)

    # ── 2. Load full AIS data ────────────────────────────────────────────────────
    logger.info("[2/4] Loading AIS data ...")
    df = AISDataLoader(args.data_path).load_data()
    logger.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

    # ── 3. Run gamma × w1 × w2 sweep ────────────────────────────────────────────
    w1_grid = [1.0] if args.gamma_only else config.NLSN_W1_VALUES
    w2_grid = [1.0] if args.gamma_only else config.NLSN_W2_VALUES
    n_combos = len(config.NLSN_GAMMA_VALUES) * len(w1_grid) * len(w2_grid)
    logger.info("[3/4] Starting gamma sweep (%d trials) ...", n_combos)

    cfg = NLSNGammaConfig(
        gamma_values         = config.NLSN_GAMMA_VALUES,
        w1_values            = w1_grid,
        w2_values            = w2_grid,
        min_trajectory_points = config.NLSN_MIN_TRAJ_POINTS,
        max_time_gap_minutes  = config.NLSN_MAX_TIME_GAP_MIN,
        alpha_shape           = config.NLSN_ALPHA_SHAPE,
        expected_nodes_min    = config.NLSN_EXPECTED_NODES_MIN,
        expected_nodes_max    = config.NLSN_EXPECTED_NODES_MAX,
        map_sample_size       = config.VISUALIZATION_SAMPLE_SIZE,
    )

    tuner = NLSNGammaTuner(
        output_dir   = args.nlsn_dir,
        require_cuda = use_cuda,
        show_progress= True,
    )
    results_df = tuner.run_sweep(df, plsn_params, cfg)
    logger.info("  Sweep done: %d trials recorded.", len(results_df))

    # ── 4. Print summary ─────────────────────────────────────────────────────────
    logger.info("[4/4] Pipeline complete.")
    best = results_df.iloc[0] if not results_df.empty else None
    sep  = "-" * 70
    print(f"\n{sep}")
    print("  NLSN Full Pipeline Complete")
    print(sep)
    if best is not None:
        print(f"  Best gamma  : {best['gamma']:.6g}  w1={best['w1']:.1f}  w2={best['w2']:.1f}")
        print(f"  Score       : {best['score']:.4f}")
        print(f"  Nodes/Edges : {int(best['nodes'])} / {int(best['edges'])}")
        print(f"  LD / Dr / Dl: {best['ld_score']:.4f} / {best['compression_rate_dr']:.4f} / {best['distance_similarity_dl']:.4f}")
    comparison_html = os.path.join(args.nlsn_dir, "nlsn_gamma_comparison.html")
    print(f"  Dashboard   : {comparison_html}")
    print(f"  Log         : {log_path}")
    print(sep)


if __name__ == "__main__":
    main()
