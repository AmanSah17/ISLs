import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run NLSN gamma sweep from Jan-2025 best PLSN outputs.")
    parser.add_argument(
        "--output-root",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results",
    )
    parser.add_argument(
        "--python-bin",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/code_/env/bin/python",
    )
    parser.add_argument(
        "--gamma-values",
        default="0.00005,0.00008,0.0001,0.00015,0.0002,0.0003,0.0004",
    )
    parser.add_argument("--min-trajectory-points", type=int, default=5)
    parser.add_argument("--max-time-gap-minutes", type=float, default=720.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--expected-nodes-min", type=int, default=None)
    parser.add_argument("--expected-nodes-max", type=int, default=None)
    parser.add_argument("--map-sample-size", type=int, default=120000)
    parser.add_argument("--max-input-rows", type=int, default=5_000_000)
    parser.add_argument("--sample-mmsi-count", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_root = args.output_root
    full_data = os.path.join(output_root, "data", "jan2025_full_minimal.parquet")
    plsn_summary = os.path.join(output_root, "tuning", "expanded_hyperparam_sweep_summary.json")
    plsn_results = os.path.join(output_root, "tuning", "expanded_hyperparam_sweep_results.csv")
    nlsn_output = os.path.join(output_root, "nlsn_gamma_sweep")

    if not os.path.exists(full_data):
        raise FileNotFoundError(f"Full AIS parquet not found: {full_data}")

    cmd = [
        args.python_bin,
        "CEE_From_ports_to_routes:extracting_maritime_traffic_patterns/NLSN_Extraction/run_nlsn_gamma_sweep.py",
        "--full-data-path",
        full_data,
        "--output-dir",
        nlsn_output,
        "--plsn-summary-json",
        plsn_summary,
        "--plsn-results-csv",
        plsn_results,
        "--gamma-values",
        args.gamma_values,
        "--min-trajectory-points",
        str(args.min_trajectory_points),
        "--max-time-gap-minutes",
        str(args.max_time_gap_minutes),
        "--w1",
        str(args.w1),
        "--w2",
        str(args.w2),
        "--map-sample-size",
        str(args.map_sample_size),
        "--max-input-rows",
        str(args.max_input_rows),
        "--sample-mmsi-count",
        str(args.sample_mmsi_count),
    ]
    if args.expected_nodes_min is not None:
        cmd.extend(["--expected-nodes-min", str(args.expected_nodes_min)])
    if args.expected_nodes_max is not None:
        cmd.extend(["--expected-nodes-max", str(args.expected_nodes_max)])
    if args.require_cuda:
        cmd.append("--require-cuda")
    if args.no_progress:
        cmd.append("--no-progress")

    run_cmd(cmd)
    print("\nNLSN workflow completed.")
    print("Output directory:", nlsn_output)
    print("Comparison HTML:", os.path.join(nlsn_output, "nlsn_gamma_comparison.html"))


if __name__ == "__main__":
    main()
