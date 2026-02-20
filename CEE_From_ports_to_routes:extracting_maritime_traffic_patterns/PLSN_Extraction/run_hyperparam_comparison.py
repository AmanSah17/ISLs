import json
import logging
import os
import sys
import argparse
from datetime import datetime

import pandas as pd
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from PLSN_Extraction import config
from PLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
from PLSN_Extraction.modules.clustering import CLIQUEClusterer
from PLSN_Extraction.modules.data_loader import AISDataLoader
from PLSN_Extraction.modules.hyperparameter_tuner import CLIQUEHyperparameterTuner, TuningConfig
from PLSN_Extraction.modules.network_generator import PLSNGenerator
from PLSN_Extraction.modules.preprocessor import AISPreprocessor
from PLSN_Extraction.modules.visualizer import PLSNVisualizer


class _DropNoisyWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Singular matrix. Likely caused by all points lying in an N-1 space." in msg:
            return False
        return True


def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    # Suppress extremely chatty libraries that can overwhelm terminal/IDE buffers.
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.cuda").setLevel(logging.WARNING)
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
    logging.getLogger("pyogrio").setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropNoisyWarnings())


def method_id(row: pd.Series) -> str:
    r_str = str(row["r"]).replace(".", "p")
    return f"k{int(row['k'])}_r{r_str}_n{row['neighbor_mode']}"


def build_comparison_html(
    output_path: str,
    all_results: pd.DataFrame,
    selected_methods: list[dict],
):
    all_rows_html = "".join(
        [
            "<tr>"
            f"<td>{int(r['trial'])}</td>"
            f"<td>{int(r['k'])}</td>"
            f"<td>{r['r']}</td>"
            f"<td>{r['neighbor_mode']}</td>"
            f"<td>{r['min_dense_points']}</td>"
            f"<td>{int(r['n_clusters'])}</td>"
            f"<td>{int(r['effective_ports'])}</td>"
            f"<td>{int(r['edge_count'])}</td>"
            f"<td>{int(r['transition_count'])}</td>"
            f"<td>{r['coverage_ratio']:.4f}</td>"
            f"<td>{r['noise_ratio']:.4f}</td>"
            f"<td>{r['largest_cluster_share']:.4f}</td>"
            f"<td>{r['runtime_sec']:.3f}</td>"
            f"<td>{r['score']:.4f}</td>"
            "</tr>"
            for _, r in all_results.iterrows()
        ]
    )

    selected_rows_html = "".join(
        [
            "<tr>"
            f"<td>{m['method_name']}</td>"
            f"<td>{int(m['k'])}</td>"
            f"<td>{m['r']}</td>"
            f"<td>{m['neighbor_mode']}</td>"
            f"<td>{m['ports']}</td>"
            f"<td>{m['edges']}</td>"
            f"<td>{m['transitions']}</td>"
            f"<td>{m['score']:.4f}</td>"
            f"<td><a href='{m['map_rel_path']}' target='_blank'>Open Map</a></td>"
            "</tr>"
            for m in selected_methods
        ]
    )

    default_map = selected_methods[0]["map_rel_path"] if selected_methods else ""
    methods_json = json.dumps(selected_methods)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PLSN Hyperparameter Comparison</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, sans-serif; background: #f8fafc; color: #111827; }}
    .wrap {{ padding: 16px; }}
    h1, h2 {{ margin: 8px 0; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px; text-align: left; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    .scroll {{ max-height: 360px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
    .row {{ display: grid; grid-template-columns: 1fr 2fr; gap: 12px; }}
    iframe {{ width: 100%; height: 720px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; }}
    select {{ width: 100%; padding: 8px; font-size: 14px; }}
    .muted {{ color: #6b7280; font-size: 12px; }}
    @media (max-width: 1000px) {{ .row {{ grid-template-columns: 1fr; }} iframe {{ height: 520px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PLSN Hyperparameter Comparison Dashboard</h1>
    <p class="muted">Generated: {datetime.now().isoformat(timespec='seconds')}</p>

    <div class="card">
      <h2>Top Methods (Detailed Maps)</h2>
      <div class="row">
        <div>
          <label for="method-select"><b>Select method map:</b></label>
          <select id="method-select"></select>
          <p class="muted">Each map shows extracted port regions, nodes, lanes/edges, and traffic-density context.</p>
          <div class="scroll">
            <table>
              <thead>
                <tr><th>Method</th><th>K</th><th>r</th><th>Nbr</th><th>Ports</th><th>Edges</th><th>Transitions</th><th>Score</th><th>Map</th></tr>
              </thead>
              <tbody>
                {selected_rows_html}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <iframe id="map-frame" src="{default_map}"></iframe>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>All Hyperparameter Trials</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Trial</th><th>K</th><th>r</th><th>Nbr</th><th>MinDense</th><th>Clusters</th>
              <th>Ports</th><th>Edges</th><th>Transitions</th><th>Coverage</th><th>Noise</th>
              <th>LargestShare</th><th>Runtime(s)</th><th>Score</th>
            </tr>
          </thead>
          <tbody>
            {all_rows_html}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const methods = {methods_json};
    const select = document.getElementById('method-select');
    const frame = document.getElementById('map-frame');
    methods.forEach((m, idx) => {{
      const opt = document.createElement('option');
      opt.value = m.map_rel_path;
      opt.textContent = `${{idx+1}}. ${{m.method_name}} (ports=${{m.ports}}, edges=${{m.edges}})`;
      select.appendChild(opt);
    }});
    select.addEventListener('change', () => {{
      frame.src = select.value;
    }});
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Run expanded CLIQUE hyperparameter comparison and map dashboard.")
    parser.add_argument("--data-path", type=str, default=config.DATA_FILE_PATH)
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--top-maps", type=int, default=config.TUNING_TOP_MAPS)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--memory-cap-mb", type=float, default=4096.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "hyperparam_comparison.log")
    setup_logging(log_path)
    logger = logging.getLogger("Hyperparam-Comparison")

    logger.info("Loading data...")
    df = AISDataLoader(args.data_path).load_data()
    stationary_df = AISPreprocessor(
        sog_threshold=config.SOG_THRESHOLD,
        nav_status_filter=config.NAV_STATUS_FILTER,
    ).filter_anchor_mooring(df)
    logger.info("Stationary candidate rows: %d", len(stationary_df))

    tuning_cfg = TuningConfig(
        k_values=config.TUNING_K_VALUES,
        r_values=config.TUNING_R_VALUES,
        neighbor_modes=config.TUNING_NEIGHBOR_MODES,
        min_dense_points_values=config.TUNING_MIN_DENSE_POINTS,
        min_port_points=config.TUNING_MIN_PORT_POINTS,
        expected_ports_min=config.TUNING_EXPECTED_PORTS_MIN,
        expected_ports_max=config.TUNING_EXPECTED_PORTS_MAX,
    )

    tuner = CLIQUEHyperparameterTuner(args.output_dir, require_cuda=True, show_progress=True)
    checkpoint_path = args.checkpoint_path.strip() or os.path.join(args.output_dir, "expanded_hyperparam_sweep_checkpoint.csv")
    results_df = tuner.run_sweep(
        stationary_df,
        tuning_cfg,
        checkpoint_path=checkpoint_path,
        resume=args.resume_from_checkpoint,
        memory_cap_mb=args.memory_cap_mb if args.memory_cap_mb > 0 else None,
    )
    results_csv, results_summary = tuner.export_results(results_df, prefix="expanded_hyperparam_sweep")
    logger.info("Sweep exported: %s | %s", results_csv, results_summary)

    # Build detailed maps for top scoring methods.
    top_n = min(args.top_maps, len(results_df))
    selected = results_df.head(top_n).copy()
    comparison_root = os.path.join(args.output_dir, "hyperparam_maps")
    os.makedirs(comparison_root, exist_ok=True)

    method_rows: list[dict] = []
    lat_col = "LAT" if "LAT" in df.columns else "Latitude"
    lon_col = "LON" if "LON" in df.columns else "Longitude"

    # Reduce map payload while keeping global context.
    map_full_df = df[[lat_col, lon_col]].dropna()
    if len(map_full_df) > 400000:
        map_full_df = map_full_df.sample(n=400000, random_state=42)

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Detailed map generation", unit="map"):
        mname = method_id(row)
        logger.info("Generating detailed map for method: %s", mname)
        method_dir = os.path.join(comparison_root, mname)
        os.makedirs(method_dir, exist_ok=True)

        clusterer = CLIQUEClusterer(
            k=int(row["k"]),
            density_threshold_r=float(row["r"]),
            min_dense_points=None if pd.isna(row["min_dense_points"]) else int(row["min_dense_points"]),
            neighbor_mode=str(row["neighbor_mode"]),
            require_cuda=True,
        )
        clustered_df = clusterer.fit_predict(stationary_df)

        boundaries = BoundaryExtractor(alpha=config.ALPHA_SHAPE_PARAMETER).extract_boundaries(clustered_df)
        generator = PLSNGenerator(output_dir=method_dir)
        nodes_df = generator.export_nodes_and_boundaries(boundaries, clustered_df=clustered_df)
        if nodes_df is None or nodes_df.empty:
            nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "area_deg2", "stationary_points"])
        valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
        edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)

        map_path = os.path.join(method_dir, "plsn_map.html")
        visualizer = PLSNVisualizer(output_file=map_path, sample_size=80000)
        visualizer.generate_plsn_dashboard(map_full_df, clustered_df, boundaries, nodes_df, edges_df)

        method_rows.append(
            {
                "method_name": mname,
                "k": int(row["k"]),
                "r": float(row["r"]),
                "neighbor_mode": str(row["neighbor_mode"]),
                "ports": int(row["effective_ports"]),
                "edges": int(row["edge_count"]),
                "transitions": int(row["transition_count"]),
                "score": float(row["score"]),
                "map_rel_path": os.path.relpath(map_path, args.output_dir),
            }
        )

    comparison_html = os.path.join(args.output_dir, "hyperparam_comparison_map.html")
    build_comparison_html(comparison_html, results_df, method_rows)
    logger.info("Comparison HTML written to: %s", comparison_html)


if __name__ == "__main__":
    main()
