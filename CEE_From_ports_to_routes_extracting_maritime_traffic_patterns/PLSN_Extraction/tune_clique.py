import argparse
from datetime import datetime
import logging
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from PLSN_Extraction import config
from PLSN_Extraction.modules.data_loader import AISDataLoader
from PLSN_Extraction.modules.preprocessor import AISPreprocessor
from PLSN_Extraction.modules.hyperparameter_tuner import (
    CLIQUEHyperparameterTuner,
    TuningConfig,
)


class _DropNoisyWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Singular matrix. Likely caused by all points lying in an N-1 space." in msg:
            return False
        return True


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_mode_list(value: str) -> list[str]:
    modes = [v.strip() for v in value.split(",") if v.strip()]
    for mode in modes:
        if mode not in {"4", "8"}:
            raise ValueError(f"Invalid neighbor mode: {mode}. Allowed: 4,8")
    return modes


def _parse_optional_int_list(value: str) -> list[int | None]:
    out: list[int | None] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(token))
    return out


def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.cuda").setLevel(logging.WARNING)
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropNoisyWarnings())


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for PLSN CLIQUE clustering.")
    parser.add_argument("--data-path", type=str, default=config.DATA_FILE_PATH)
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of stationary points used for tuning.")
    parser.add_argument("--sample-max", type=int, default=0, help="Optional hard cap for sampled points (0=disabled).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(v) for v in config.TUNING_K_VALUES),
    )
    parser.add_argument(
        "--r-values",
        type=str,
        default=",".join(str(v) for v in config.TUNING_R_VALUES),
    )
    parser.add_argument(
        "--neighbor-modes",
        type=str,
        default=",".join(config.TUNING_NEIGHBOR_MODES),
    )
    parser.add_argument(
        "--min-dense-points",
        type=str,
        default=",".join("none" if v is None else str(v) for v in config.TUNING_MIN_DENSE_POINTS),
    )
    parser.add_argument("--min-port-points", type=int, default=config.TUNING_MIN_PORT_POINTS)
    parser.add_argument("--expected-ports-min", type=int, default=config.TUNING_EXPECTED_PORTS_MIN)
    parser.add_argument("--expected-ports-max", type=int, default=config.TUNING_EXPECTED_PORTS_MAX)
    parser.add_argument("--run-name", type=str, default="", help="Optional output prefix for this tuning run.")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--memory-cap-mb", type=float, default=4096.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "clique_tuning.log")
    setup_logging(log_path)
    logger = logging.getLogger("CLIQUE-Tuner")

    logger.info("Loading AIS data from %s", args.data_path)
    loader = AISDataLoader(args.data_path)
    df = loader.load_data()

    preprocessor = AISPreprocessor(
        sog_threshold=config.SOG_THRESHOLD,
        nav_status_filter=config.NAV_STATUS_FILTER,
    )
    stationary_df = preprocessor.filter_anchor_mooring(df)
    logger.info("Stationary candidate points before sampling: %d", len(stationary_df))

    if args.sample_frac < 1.0:
        stationary_df = stationary_df.sample(frac=args.sample_frac, random_state=args.random_state)
        logger.info("Applied sample_frac=%.4f -> %d rows", args.sample_frac, len(stationary_df))

    if args.sample_max > 0 and len(stationary_df) > args.sample_max:
        stationary_df = stationary_df.sample(n=args.sample_max, random_state=args.random_state)
        logger.info("Applied sample_max=%d -> %d rows", args.sample_max, len(stationary_df))

    tuning_cfg = TuningConfig(
        k_values=_parse_int_list(args.k_values),
        r_values=_parse_float_list(args.r_values),
        neighbor_modes=_parse_mode_list(args.neighbor_modes),
        min_dense_points_values=_parse_optional_int_list(args.min_dense_points),
        min_port_points=args.min_port_points,
        expected_ports_min=args.expected_ports_min,
        expected_ports_max=args.expected_ports_max,
    )

    tuner = CLIQUEHyperparameterTuner(output_dir=args.output_dir, require_cuda=True, show_progress=True)
    checkpoint_path = args.checkpoint_path.strip() or os.path.join(args.output_dir, "clique_tuning_checkpoint.csv")
    results_df = tuner.run_sweep(
        stationary_df,
        tuning_cfg,
        checkpoint_path=checkpoint_path,
        resume=args.resume_from_checkpoint,
        memory_cap_mb=args.memory_cap_mb if args.memory_cap_mb > 0 else None,
    )
    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"clique_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path, summary_path = tuner.export_results(results_df, prefix=run_name)

    best = results_df.iloc[0].to_dict() if not results_df.empty else {}
    logger.info("Best trial: %s", best)
    print("\nBest parameters:")
    if best:
        print(
            f"  K={int(best['k'])}, r={best['r']}, neighbor_mode={best['neighbor_mode']}, "
            f"min_dense_points={best['min_dense_points']}, effective_ports={int(best['effective_ports'])}, "
            f"score={best['score']:.4f}"
        )
    print(f"\nResults CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
