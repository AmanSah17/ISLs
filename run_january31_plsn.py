import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import duckdb
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run 31-day January PLSN + hyperparameter tuning workflow.")
    parser.add_argument(
        "--daily-parquet-glob",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/processed_data/daily/processed_2025-01-*.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results",
    )
    parser.add_argument(
        "--python-bin",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/code_/env/bin/python",
    )
    parser.add_argument("--skip-consolidation", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--memory-cap-mb", type=float, default=4096.0)
    return parser.parse_args()


def run_cmd(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_root = args.output_root
    data_dir = os.path.join(out_root, "data")
    tuning_dir = os.path.join(out_root, "tuning")
    best_dir = os.path.join(out_root, "plsn_best")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tuning_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    full_minimal = os.path.join(data_dir, "jan2025_full_minimal.parquet")
    stationary_minimal = os.path.join(data_dir, "jan2025_stationary_anchor_moor.parquet")
    daily_full_dir = os.path.join(data_dir, "daily_full_minimal")
    daily_stationary_dir = os.path.join(data_dir, "daily_stationary_minimal")
    os.makedirs(daily_full_dir, exist_ok=True)
    os.makedirs(daily_stationary_dir, exist_ok=True)

    if not args.skip_consolidation:
        daily_files = sorted(glob.glob(args.daily_parquet_glob))
        if not daily_files:
            raise FileNotFoundError(f"No parquet files matched glob: {args.daily_parquet_glob}")

        con = duckdb.connect()
        for idx, src in enumerate(daily_files, start=1):
            day_token = os.path.basename(src).replace(".parquet", "")
            out_full = os.path.join(daily_full_dir, f"{day_token}_minimal.parquet")
            out_stat = os.path.join(daily_stationary_dir, f"{day_token}_stationary.parquet")
            print(f"[{idx}/{len(daily_files)}] Serial processing: {src}")
            con.execute(
                f"""
COPY (
  SELECT
    CAST(MMSI AS BIGINT) AS MMSI,
    CAST(BASEDATETIME AS TIMESTAMP) AS BASEDATETIME,
    CAST(LAT AS DOUBLE) AS LAT,
    CAST(LON AS DOUBLE) AS LON,
    CAST(SOG AS DOUBLE) AS SOG,
    CAST(COG AS DOUBLE) AS COG,
    CAST(NAVSTATUS AS DOUBLE) AS NAVSTATUS
  FROM read_parquet('{src}')
) TO '{out_full}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""
            )
            con.execute(
                f"""
COPY (
  SELECT
    CAST(MMSI AS BIGINT) AS MMSI,
    CAST(BASEDATETIME AS TIMESTAMP) AS BASEDATETIME,
    CAST(LAT AS DOUBLE) AS LAT,
    CAST(LON AS DOUBLE) AS LON,
    CAST(SOG AS DOUBLE) AS SOG,
    CAST(COG AS DOUBLE) AS COG,
    CAST(NAVSTATUS AS DOUBLE) AS NAVSTATUS
  FROM read_parquet('{src}')
  WHERE NAVSTATUS IN (1, 5)
    AND SOG < 0.5
    AND LAT BETWEEN -90 AND 90
    AND LON BETWEEN -180 AND 180
    AND MMSI IS NOT NULL
    AND BASEDATETIME IS NOT NULL
) TO '{out_stat}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""
            )

        con.execute(
            f"""
COPY (
  SELECT * FROM read_parquet('{daily_full_dir}/*.parquet')
) TO '{full_minimal}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""
        )
        con.execute(
            f"""
COPY (
  SELECT * FROM read_parquet('{daily_stationary_dir}/*.parquet')
) TO '{stationary_minimal}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""
        )
        con.close()
        print("Wrote:", full_minimal)
        print("Wrote:", stationary_minimal)
    elif not (os.path.exists(full_minimal) and os.path.exists(stationary_minimal)):
        raise FileNotFoundError("Skip-consolidation set, but consolidated parquet files are missing.")

    # Run expanded tuning + detailed comparison maps on stationary dataset.
    if not args.skip_tuning:
        run_cmd(
            [
                args.python_bin,
                "CEE_From_ports_to_routes:extracting_maritime_traffic_patterns/PLSN_Extraction/run_hyperparam_comparison.py",
                "--data-path",
                stationary_minimal,
                "--output-dir",
                tuning_dir,
                "--top-maps",
                "12",
                "--resume-from-checkpoint",
                "--checkpoint-path",
                os.path.join(tuning_dir, "expanded_hyperparam_sweep_checkpoint.csv"),
                "--memory-cap-mb",
                str(args.memory_cap_mb),
            ]
        )

    # Pick best tuning row.
    results_csv = os.path.join(tuning_dir, "expanded_hyperparam_sweep_results.csv")
    checkpoint_csv = os.path.join(tuning_dir, "expanded_hyperparam_sweep_checkpoint.csv")
    if not os.path.exists(results_csv):
        raise FileNotFoundError("Tuning results CSV not found. Run without --skip-tuning first.")
    results_df = pd.read_csv(results_csv)
    if results_df.empty:
        if os.path.exists(checkpoint_csv):
            ckpt_df = pd.read_csv(checkpoint_csv)
            if "score" in ckpt_df.columns:
                ckpt_df["score"] = pd.to_numeric(ckpt_df["score"], errors="coerce")
                ckpt_df = ckpt_df[ckpt_df["score"].notna()].copy()
                if not ckpt_df.empty:
                    results_df = ckpt_df.sort_values("score", ascending=False).reset_index(drop=True)
        if results_df.empty:
            raise RuntimeError(
                "No completed tuning trials were found. Increase --memory-cap-mb or rerun with "
                "--skip-consolidation to continue from existing data/checkpoint."
            )
    best = results_df.iloc[0].to_dict()
    best_k = int(best["k"])
    best_r = float(best["r"])
    best_neighbor = "4" if pd.isna(best["neighbor_mode"]) else str(int(float(best["neighbor_mode"])))
    best_min_dense = None if pd.isna(best["min_dense_points"]) else int(best["min_dense_points"])

    # Run final best-parameter PLSN and write all artifacts to plsn_best folder.
    run_best_sql = f"""
import os
import sys
import pandas as pd

repo_root = os.getcwd()
package_parent = os.path.join(repo_root, "CEE_From_ports_to_routes:extracting_maritime_traffic_patterns")
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

from PLSN_Extraction.modules.data_loader import AISDataLoader
from PLSN_Extraction.modules.preprocessor import AISPreprocessor
from PLSN_Extraction.modules.clustering import CLIQUEClusterer
from PLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
from PLSN_Extraction.modules.network_generator import PLSNGenerator
from PLSN_Extraction.modules.visualizer import PLSNVisualizer

data_path = "{stationary_minimal}"
full_path = "{full_minimal}"
output_dir = "{best_dir}"
best_k = {best_k}
best_r = {best_r}
best_neighbor = "{best_neighbor}"
best_min_dense = {repr(best_min_dense)}

os.makedirs(output_dir, exist_ok=True)

df_stationary = AISDataLoader(data_path).load_data()
df_full = AISDataLoader(full_path).load_data()

pre = AISPreprocessor(sog_threshold=0.5, nav_status_filter=[1, 5])
stationary_df = pre.filter_anchor_mooring(df_stationary)

clusterer = CLIQUEClusterer(
    k=best_k,
    density_threshold_r=best_r,
    min_dense_points=best_min_dense,
    neighbor_mode=best_neighbor,
    require_cuda=True,
)
clustered_df = clusterer.fit_predict(stationary_df)

extractor = BoundaryExtractor(alpha=0.01)
boundaries = extractor.extract_boundaries(clustered_df)

generator = PLSNGenerator(output_dir=output_dir)
nodes_df = generator.export_nodes_and_boundaries(boundaries, clustered_df=clustered_df)
if nodes_df is None or nodes_df.empty:
    nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "area_deg2", "stationary_points"])
valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)

vis = PLSNVisualizer(output_file=os.path.join(output_dir, "plsn_map_best.html"), sample_size=500000)
vis.generate_plsn_dashboard(df_full, clustered_df, boundaries, nodes_df, edges_df)

summary = {{
    "data_path_stationary": data_path,
    "data_path_full": full_path,
    "best_params": {{
        "k": best_k,
        "r": best_r,
        "neighbor_mode": best_neighbor,
        "min_dense_points": best_min_dense,
    }},
    "ports_extracted": int(len(nodes_df)),
    "edges_extracted": int(len(edges_df)),
    "clustered_points": int((clustered_df["cluster_id"] != -1).sum()),
    "noise_points": int((clustered_df["cluster_id"] == -1).sum()),
}}
import json
with open(os.path.join(output_dir, "best_run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
"""
    run_cmd([args.python_bin, "-c", run_best_sql])

    # Global run summary.
    run_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": out_root,
        "consolidated_full": full_minimal,
        "consolidated_stationary": stationary_minimal,
        "tuning_results_csv": results_csv,
        "comparison_html": os.path.join(tuning_dir, "hyperparam_comparison_map.html"),
        "best_output_dir": best_dir,
        "memory_cap_mb": args.memory_cap_mb,
        "best_params": {
            "k": best_k,
            "r": best_r,
            "neighbor_mode": best_neighbor,
            "min_dense_points": best_min_dense,
        },
    }
    summary_path = os.path.join(out_root, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\nWorkflow completed.")
    print("Run summary:", summary_path)


if __name__ == "__main__":
    main()
