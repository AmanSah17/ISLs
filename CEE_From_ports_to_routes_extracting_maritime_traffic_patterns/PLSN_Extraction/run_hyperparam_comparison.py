"""
Enhanced PLSN hyperparameter comparison runner.

Changes vs original:
  • Sortable all-trials table (click any column header)
  • Inline Chart.js panels: bar-chart of scores, K vs effective_ports scatter,
    r vs coverage_ratio line chart
  • Per-parameter influence summary (avg score grouped by K, r, neighbor_mode)
  • Summary cards: total trials, best score, best K, best r
  • Colour-coded score cells (green=high, red=low)
  • CSV download button
  • require_cuda=False so it runs on CPU-only machines
  • Saves all outputs under results/ organised folder structure
"""

from __future__ import annotations

import json
import logging
import os
import sys
import argparse
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
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


# ── Logging helpers ────────────────────────────────────────────────────────────
class _DropNoisyWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Singular matrix" not in record.getMessage()


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    for lib in ("numba", "numba.cuda", "numba.cuda.cudadrv.driver", "pyogrio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropNoisyWarnings())


# ── Helpers ────────────────────────────────────────────────────────────────────
def method_id(row: pd.Series) -> str:
    r_str = str(row["r"]).replace(".", "p")
    return f"k{int(row['k'])}_r{r_str}_n{row['neighbor_mode']}"


def _score_cell_style(score: float, min_s: float, max_s: float) -> str:
    """Return an inline background colour for a score cell (green=high, red=low)."""
    if max_s <= min_s:
        return ""
    t = (score - min_s) / (max_s - min_s)
    # interpolate red(255,59,59) → green(34,197,94) via t
    r = int(255 + t * (34  - 255))
    g = int(59  + t * (197 - 59))
    b = int(59  + t * (94  - 59))
    return f"background:rgba({r},{g},{b},0.25);font-weight:600;"


# ── Comparison HTML builder ────────────────────────────────────────────────────
def build_comparison_html(
    output_path: str,
    all_results: pd.DataFrame,
    selected_methods: list[dict],
    generated_at: str = "",
) -> None:
    if all_results.empty:
        return

    score_min = float(all_results["score"].min())
    score_max = float(all_results["score"].max())

    # ── All-trials table rows ──────────────────────────────────────────────────
    all_row_parts = []
    for _, r in all_results.iterrows():
        score_val = float(r["score"])
        score_style = _score_cell_style(score_val, score_min, score_max)
        all_row_parts.append(
            "<tr>"
            f"<td>{int(r['trial'])}</td>"
            f"<td>{int(r['k'])}</td>"
            f"<td>{r['r']:.6f}</td>"
            f"<td>{r['neighbor_mode']}</td>"
            f"<td>{r['min_dense_points']}</td>"
            f"<td>{int(r['n_clusters'])}</td>"
            f"<td>{int(r['effective_ports'])}</td>"
            f"<td>{int(r['edge_count'])}</td>"
            f"<td>{int(r['transition_count'])}</td>"
            f"<td>{r['coverage_ratio']:.4f}</td>"
            f"<td>{r['noise_ratio']:.4f}</td>"
            f"<td>{r['largest_cluster_share']:.4f}</td>"
            f"<td>{r['runtime_sec']:.2f}s</td>"
            f"<td style='{score_style}'>{score_val:.4f}</td>"
            "</tr>"
        )
    all_rows_html = "".join(all_row_parts)

    # ── Selected methods table rows ────────────────────────────────────────────
    selected_row_parts = []
    for m in selected_methods:
        m_score = float(m["score"])
        m_style = _score_cell_style(m_score, score_min, score_max)
        selected_row_parts.append(
            "<tr>"
            f"<td><code>{m['method_name']}</code></td>"
            f"<td>{int(m['k'])}</td>"
            f"<td>{m['r']:.6f}</td>"
            f"<td>{m['neighbor_mode']}</td>"
            f"<td>{m['ports']}</td>"
            f"<td>{m['edges']}</td>"
            f"<td>{m['transitions']:,}</td>"
            f"<td style='{m_style}'>{m_score:.4f}</td>"
            f"<td><a class='map-link' href='{m['map_rel_path']}' target='_blank'>&#128506; Open</a></td>"
            "</tr>"
        )
    selected_rows_html = "".join(selected_row_parts)

    default_map  = selected_methods[0]["map_rel_path"] if selected_methods else ""
    methods_json = json.dumps(selected_methods)

    # ── Chart data ─────────────────────────────────────────────────────────────
    # (a) Score per trial — bar chart
    trial_labels  = all_results["trial"].astype(str).tolist()
    trial_scores  = all_results["score"].round(4).tolist()

    # (b) K vs effective_ports scatter coloured by score
    k_vals        = all_results["k"].tolist()
    port_vals     = all_results["effective_ports"].astype(int).tolist()
    score_vals    = all_results["score"].round(4).tolist()

    # (c) r vs coverage_ratio line chart (aggregate mean by r)
    r_group       = all_results.groupby("r")["coverage_ratio"].mean().reset_index()
    r_labels      = r_group["r"].apply(lambda v: f"{v:.6f}").tolist()
    r_cov_vals    = r_group["coverage_ratio"].round(4).tolist()

    # ── Per-parameter influence tables ─────────────────────────────────────────
    def _influence_rows(df: pd.DataFrame, col: str, label_fn=str) -> str:
        grp = df.groupby(col)["score"].agg(["mean", "max", "count"]).reset_index()
        grp = grp.sort_values("mean", ascending=False)
        return "".join(
            f"<tr><td>{label_fn(row[col])}</td><td>{row['mean']:.4f}</td>"
            f"<td>{row['max']:.4f}</td><td>{int(row['count'])}</td></tr>"
            for _, row in grp.iterrows()
        )

    k_inf_rows    = _influence_rows(all_results, "k")
    r_inf_rows    = _influence_rows(all_results, "r", label_fn=lambda v: f"{v:.6f}")
    nbr_inf_rows  = _influence_rows(all_results, "neighbor_mode")

    # ── Best trial summary ─────────────────────────────────────────────────────
    best          = all_results.iloc[0]
    total_trials  = len(all_results)

    # ── CSV download data ──────────────────────────────────────────────────────
    csv_data_json = all_results.to_json(orient="records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PLSN Hyperparameter Comparison Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0f172a; color:#f1f5f9; }}
    .wrap {{ padding:20px; max-width:1800px; margin:0 auto; }}
    h1 {{ font-size:22px; color:#38bdf8; }}
    .meta {{ font-size:11px; color:#64748b; margin:4px 0 18px; }}
    /* Summary cards */
    .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:20px; }}
    .card {{
      flex:1; min-width:120px; background:#1e293b;
      border:1px solid #334155; border-radius:12px;
      padding:14px 18px; text-align:center;
    }}
    .card .v {{ font-size:24px; font-weight:700; color:#f8fafc; }}
    .card .l {{ font-size:11px; color:#94a3b8; margin-top:4px; }}
    .card.best {{ border-color:#38bdf8; }}
    /* Section header */
    .sec {{ font-size:16px; font-weight:700; color:#93c5fd; margin:20px 0 10px; border-bottom:1px solid #1e3a5f; padding-bottom:6px; }}
    /* Tables */
    .scroll {{ max-height:400px; overflow:auto; border-radius:8px; border:1px solid #1e3a5f; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th {{
      background:#1e293b; color:#94a3b8; padding:8px 6px;
      text-align:left; font-size:11px; position:sticky; top:0; z-index:1;
      cursor:pointer; user-select:none;
      border-bottom:2px solid #334155;
    }}
    th:hover {{ background:#273548; }}
    th.asc::after  {{ content:' ▲'; color:#38bdf8; }}
    th.desc::after {{ content:' ▼'; color:#38bdf8; }}
    td {{ padding:6px; border-bottom:1px solid #1e293b; }}
    tr:hover td {{ background:rgba(56,189,248,0.07); }}
    .map-link {{ color:#38bdf8; text-decoration:none; font-weight:600; }}
    .map-link:hover {{ text-decoration:underline; }}
    /* Charts grid */
    .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; margin-bottom:20px; }}
    .chart-box {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; }}
    .chart-box h3 {{ font-size:13px; color:#93c5fd; margin-bottom:10px; }}
    canvas {{ width:100%!important; max-height:220px!important; }}
    /* Influence grid */
    .inf-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin-bottom:20px; }}
    .inf-box {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:14px; }}
    .inf-box h3 {{ font-size:12px; color:#93c5fd; margin-bottom:8px; }}
    .inf-box table {{ font-size:11px; }}
    /* Map iframe section */
    .map-section {{ display:grid; grid-template-columns:380px 1fr; gap:16px; margin-bottom:20px; }}
    @media(max-width:1000px) {{ .map-section {{ grid-template-columns:1fr; }} }}
    iframe {{ width:100%; height:680px; border:1px solid #334155; border-radius:10px; background:#000; }}
    select {{
      width:100%; padding:10px; border-radius:8px; font-size:13px;
      background:#1e293b; color:#f1f5f9; border:1px solid #334155; margin-bottom:10px;
    }}
    .dl-btn {{
      display:inline-block; padding:8px 16px; background:#1d4ed8; color:#fff;
      border-radius:8px; font-size:12px; cursor:pointer; border:none; margin-top:8px;
    }}
    .dl-btn:hover {{ background:#2563eb; }}
    code {{ font-size:10px; background:#0f172a; padding:1px 4px; border-radius:3px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>PLSN Hyperparameter Comparison Dashboard</h1>
  <div class="meta">Generated: {generated_at or datetime.now().isoformat(timespec='seconds')} &nbsp;|&nbsp; {total_trials} trials</div>

  <!-- ── Summary cards ──────────────────────────────────────── -->
  <div class="cards">
    <div class="card"><div class="v">{total_trials}</div><div class="l">Total Trials</div></div>
    <div class="card best"><div class="v">{best['score']:.4f}</div><div class="l">Best Score</div></div>
    <div class="card best"><div class="v">K={int(best['k'])}</div><div class="l">Best K</div></div>
    <div class="card best"><div class="v">r={best['r']:.5f}</div><div class="l">Best r</div></div>
    <div class="card best"><div class="v">{best['neighbor_mode']}</div><div class="l">Best Nbr Mode</div></div>
    <div class="card"><div class="v">{int(best['effective_ports'])}</div><div class="l">Best Ports</div></div>
    <div class="card"><div class="v">{int(best['edge_count'])}</div><div class="l">Best Edges</div></div>
    <div class="card"><div class="v">{best['coverage_ratio']:.3f}</div><div class="l">Best Coverage</div></div>
  </div>

  <!-- ── Charts ──────────────────────────────────────────────── -->
  <div class="sec">📊 Hyperparameter Influence Charts</div>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Trial Scores (sorted by rank)</h3>
      <canvas id="scoreChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>K vs Effective Ports (coloured by score)</h3>
      <canvas id="kPortsChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Density r vs Coverage Ratio</h3>
      <canvas id="rCovChart"></canvas>
    </div>
  </div>

  <!-- ── Parameter influence tables ──────────────────────────── -->
  <div class="sec">🔍 Per-Parameter Influence (avg score)</div>
  <div class="inf-grid">
    <div class="inf-box">
      <h3>K — Grid Divisions</h3>
      <table><thead><tr><th>K</th><th>Avg Score</th><th>Max Score</th><th>Trials</th></tr></thead>
      <tbody>{k_inf_rows}</tbody></table>
    </div>
    <div class="inf-box">
      <h3>r — Density Threshold</h3>
      <table><thead><tr><th>r</th><th>Avg Score</th><th>Max Score</th><th>Trials</th></tr></thead>
      <tbody>{r_inf_rows}</tbody></table>
    </div>
    <div class="inf-box">
      <h3>Neighbour Mode</h3>
      <table><thead><tr><th>Mode</th><th>Avg Score</th><th>Max Score</th><th>Trials</th></tr></thead>
      <tbody>{nbr_inf_rows}</tbody></table>
    </div>
  </div>

  <!-- ── Top-N map comparison ────────────────────────────────── -->
  <div class="sec">🗺 Top Methods — Interactive Maps</div>
  <div class="map-section">
    <div>
      <select id="method-select"></select>
      <div class="scroll">
        <table>
          <thead><tr>
            <th>Method</th><th>K</th><th>r</th><th>Nbr</th>
            <th>Ports</th><th>Edges</th><th>Trans.</th><th>Score</th><th>Map</th>
          </tr></thead>
          <tbody>{selected_rows_html}</tbody>
        </table>
      </div>
    </div>
    <div>
      <iframe id="map-frame" src="{default_map}" allowfullscreen></iframe>
    </div>
  </div>

  <!-- ── All trials table ────────────────────────────────────── -->
  <div class="sec">📋 All Hyperparameter Trials</div>
  <button class="dl-btn" onclick="downloadCSV()">⬇ Download CSV</button>
  <div class="scroll" style="margin-top:8px;">
    <table id="all-table">
      <thead><tr>
        <th onclick="sortTable(0)">Trial</th>
        <th onclick="sortTable(1)">K</th>
        <th onclick="sortTable(2)">r</th>
        <th onclick="sortTable(3)">Nbr</th>
        <th onclick="sortTable(4)">MinDense</th>
        <th onclick="sortTable(5)">Clusters</th>
        <th onclick="sortTable(6)">Ports</th>
        <th onclick="sortTable(7)">Edges</th>
        <th onclick="sortTable(8)">Transitions</th>
        <th onclick="sortTable(9)">Coverage</th>
        <th onclick="sortTable(10)">Noise</th>
        <th onclick="sortTable(11)">LrgShare</th>
        <th onclick="sortTable(12)">Runtime</th>
        <th onclick="sortTable(13)">Score</th>
      </tr></thead>
      <tbody id="all-tbody">{all_rows_html}</tbody>
    </table>
  </div>

</div><!-- .wrap -->

<script>
// ── Data from Python ─────────────────────────────────────────────────────────
const methods      = {methods_json};
const trialLabels  = {json.dumps(trial_labels)};
const trialScores  = {json.dumps(trial_scores)};
const kVals        = {json.dumps(k_vals)};
const portVals     = {json.dumps(port_vals)};
const scoreVals    = {json.dumps(score_vals)};
const rLabels      = {json.dumps(r_labels)};
const rCovVals     = {json.dumps(r_cov_vals)};
const csvData      = {csv_data_json};

// ── Method dropdown ──────────────────────────────────────────────────────────
const select = document.getElementById('method-select');
const frame  = document.getElementById('map-frame');
methods.forEach((m, idx) => {{
  const opt = document.createElement('option');
  opt.value = m.map_rel_path;
  opt.textContent = `${{idx+1}}. ${{m.method_name}} | ports=${{m.ports}} edges=${{m.edges}} score=${{m.score.toFixed(4)}}`;
  select.appendChild(opt);
}});
select.addEventListener('change', () => {{ frame.src = select.value; }});

// ── Score colour helper ──────────────────────────────────────────────────────
function scoreColor(s) {{
  const mn = Math.min(...trialScores), mx = Math.max(...trialScores);
  const t  = mx > mn ? (s - mn) / (mx - mn) : 0.5;
  const r  = Math.round(255 + t * (34  - 255));
  const g  = Math.round(59  + t * (197 - 59));
  const b  = Math.round(59  + t * (94  - 59));
  return `rgba(${{r}},${{g}},${{b}},0.7)`;
}}

// ── Chart.js charts ──────────────────────────────────────────────────────────
const chartBase = {{
  responsive:true, maintainAspectRatio:true, animation:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{ticks:{{color:'#94a3b8',font:{{size:9}}}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
    y:{{ticks:{{color:'#94a3b8',font:{{size:9}}}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
  }},
}};

// Bar: trial scores
new Chart(document.getElementById('scoreChart'), {{
  type:'bar',
  data:{{
    labels: trialLabels,
    datasets:[{{data:trialScores, backgroundColor:trialScores.map(s=>scoreColor(s)), borderRadius:2}}]
  }},
  options: Object.assign({{}}, chartBase, {{
    plugins:{{
      legend:{{display:false}},
      tooltip:{{callbacks:{{label:(c) => `Score: ${{c.raw}}`}}}}
    }}
  }})
}});

// Scatter: K vs ports coloured by score
new Chart(document.getElementById('kPortsChart'), {{
  type:'scatter',
  data:{{
    datasets:[{{
      data: kVals.map((k,i) => ({{x:k, y:portVals[i]}})),
      backgroundColor: scoreVals.map(s=>scoreColor(s)),
      pointRadius: 5,
      pointHoverRadius:8,
    }}]
  }},
  options: Object.assign({{}},chartBase,{{
    plugins:{{
      legend:{{display:false}},
      tooltip:{{callbacks:{{label:(c) => `K=${{c.raw.x}} ports=${{c.raw.y}} score=${{scoreVals[c.dataIndex].toFixed(4)}}`}}}}
    }},
    scales:{{
      x:{{title:{{display:true, text:'K (grid divisions)', color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
      y:{{title:{{display:true, text:'Effective Ports', color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
    }}
  }})
}});

// Line: r vs coverage
new Chart(document.getElementById('rCovChart'), {{
  type:'line',
  data:{{
    labels: rLabels,
    datasets:[{{
      data: rCovVals,
      borderColor:'#38bdf8', backgroundColor:'rgba(56,189,248,0.12)',
      pointRadius:4, pointBackgroundColor:'#38bdf8', tension:0.3, fill:true,
    }}]
  }},
  options: Object.assign({{}},chartBase,{{
    scales:{{
      x:{{title:{{display:true,text:'Density r',color:'#94a3b8'}},ticks:{{color:'#94a3b8',maxRotation:45,font:{{size:8}}}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
      y:{{title:{{display:true,text:'Avg Coverage Ratio',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
    }}
  }})
}});

// ── Sortable table ───────────────────────────────────────────────────────────
let sortCol = 13, sortDir = -1;  // default: sort by score desc
const thead = document.querySelectorAll('#all-table th');
function clearArrows() {{ thead.forEach(h => h.classList.remove('asc','desc')); }}
function sortTable(col) {{
  if (col === sortCol) {{ sortDir *= -1; }}
  else {{ sortCol = col; sortDir = 1; }}
  clearArrows();
  thead[col].classList.add(sortDir === 1 ? 'asc' : 'desc');

  const tbody = document.getElementById('all-tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    const av = a.cells[col].innerText.replace(/[^0-9.e\-]/g,'');
    const bv = b.cells[col].innerText.replace(/[^0-9.e\-]/g,'');
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return sortDir * (an - bn);
    return sortDir * av.localeCompare(bv);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
// Initial sort by score
(function() {{
  clearArrows();
  thead[13].classList.add('desc');
}})();

// ── CSV download ─────────────────────────────────────────────────────────────
function downloadCSV() {{
  if (!csvData.length) return;
  const keys = Object.keys(csvData[0]);
  const lines = [keys.join(','), ...csvData.map(r => keys.map(k => r[k]).join(','))];
  const blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'plsn_tuning_results.csv';
  a.click();
}}
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ── Main orchestrator ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PLSN hyperparameter sweep with enhanced interactive comparison dashboard."
    )
    parser.add_argument("--data-path",   type=str,   default=config.DATA_FILE_PATH)
    parser.add_argument("--output-dir",  type=str,   default=config.MAPS_DIR)
    parser.add_argument("--tuning-dir",  type=str,   default=config.TUNING_DIR)
    parser.add_argument("--top-maps",    type=int,   default=config.TUNING_TOP_MAPS)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--memory-cap-mb",   type=float, default=6144.0)
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU mode (no CUDA).")
    args = parser.parse_args()

    use_cuda = not args.no_cuda
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tuning_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.tuning_dir, f"hyperparam_comparison_{ts}.log")
    setup_logging(log_path)
    logger = logging.getLogger("Hyperparam-Comparison")

    # 1. Load & preprocess
    logger.info("Loading data from %s", args.data_path)
    df = AISDataLoader(args.data_path).load_data()
    stationary_df = AISPreprocessor(
        sog_threshold=config.SOG_THRESHOLD,
        nav_status_filter=config.NAV_STATUS_FILTER,
    ).filter_anchor_mooring(df)
    logger.info("Stationary candidate rows: %d", len(stationary_df))

    lat_col = "LAT" if "LAT" in df.columns else "Latitude"
    lon_col = "LON" if "LON" in df.columns else "Longitude"
    map_full_df = df[[lat_col, lon_col]].dropna()
    if len(map_full_df) > 400_000:
        map_full_df = map_full_df.sample(n=400_000, random_state=42)

    # 2. Hyperparameter sweep
    tuning_cfg = TuningConfig(
        k_values=config.TUNING_K_VALUES,
        r_values=config.TUNING_R_VALUES,
        neighbor_modes=config.TUNING_NEIGHBOR_MODES,
        min_dense_points_values=config.TUNING_MIN_DENSE_POINTS,
        min_port_points=config.TUNING_MIN_PORT_POINTS,
        expected_ports_min=config.TUNING_EXPECTED_PORTS_MIN,
        expected_ports_max=config.TUNING_EXPECTED_PORTS_MAX,
    )
    tuner = CLIQUEHyperparameterTuner(
        args.tuning_dir, require_cuda=use_cuda, show_progress=True
    )
    checkpoint_path = (
        args.checkpoint_path.strip()
        or os.path.join(args.tuning_dir, "hyperparam_checkpoint.csv")
    )
    results_df = tuner.run_sweep(
        stationary_df,
        tuning_cfg,
        checkpoint_path=checkpoint_path,
        resume=args.resume_from_checkpoint,
        memory_cap_mb=args.memory_cap_mb if args.memory_cap_mb > 0 else None,
    )
    csv_path, summary_path = tuner.export_results(results_df, prefix=f"plsn_tuning_{ts}")
    logger.info("Sweep done → %s | %s", csv_path, summary_path)

    if results_df.empty:
        logger.error("No tuning results. Exiting.")
        return

    # 3. Build detailed maps for top-N
    top_n = min(args.top_maps, len(results_df))
    selected = results_df.head(top_n).copy()
    top_maps_dir = os.path.join(args.output_dir, f"top_{top_n}_{ts}")
    os.makedirs(top_maps_dir, exist_ok=True)

    method_rows: list[dict] = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Map generation", unit="map"):
        mname = method_id(row)
        logger.info("Generating map: %s", mname)
        method_dir = os.path.join(top_maps_dir, mname)
        os.makedirs(method_dir, exist_ok=True)

        clusterer = CLIQUEClusterer(
            k=int(row["k"]),
            density_threshold_r=float(row["r"]),
            min_dense_points=None if pd.isna(row["min_dense_points"]) else int(row["min_dense_points"]),
            neighbor_mode=str(row["neighbor_mode"]),
            require_cuda=use_cuda,
        )
        clustered_df = clusterer.fit_predict(stationary_df)

        boundaries = BoundaryExtractor(alpha=config.ALPHA_SHAPE_PARAMETER).extract_boundaries(clustered_df)
        generator  = PLSNGenerator(output_dir=method_dir)
        nodes_df   = generator.export_nodes_and_boundaries(boundaries, clustered_df=clustered_df)
        if nodes_df is None or nodes_df.empty:
            nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "area_deg2", "stationary_points"])
        valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
        edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)

        params_meta = {
            "k": int(row["k"]),
            "r": float(row["r"]),
            "neighbor_mode": str(row["neighbor_mode"]),
            "min_dense_points": None if pd.isna(row["min_dense_points"]) else int(row["min_dense_points"]),
        }
        map_path  = os.path.join(method_dir, "plsn_map.html")
        visualizer = PLSNVisualizer(
            output_file=map_path,
            sample_size=config.VISUALIZATION_SAMPLE_SIZE,
            edge_min_width=config.EDGE_MIN_WIDTH,
            edge_max_width=config.EDGE_MAX_WIDTH,
            boundary_weight=config.BOUNDARY_WEIGHT,
            node_min_radius=config.NODE_MIN_RADIUS,
            node_max_radius=config.NODE_MAX_RADIUS,
        )
        visualizer.generate_plsn_dashboard(
            map_full_df, clustered_df, boundaries, nodes_df, edges_df,
            params_meta=params_meta, trial_score=float(row["score"]),
        )

        method_rows.append({
            "method_name": mname,
            "k": int(row["k"]),
            "r": float(row["r"]),
            "neighbor_mode": str(row["neighbor_mode"]),
            "ports": int(row["effective_ports"]),
            "edges": int(row["edge_count"]),
            "transitions": int(row["transition_count"]),
            "score": float(row["score"]),
            "map_rel_path": os.path.relpath(map_path, args.output_dir),
        })

    # 4. Build comparison dashboard
    comparison_html = os.path.join(args.output_dir, f"plsn_comparison_{ts}.html")
    build_comparison_html(
        comparison_html,
        results_df,
        method_rows,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    logger.info("Comparison dashboard -> %s", comparison_html)
    print(f"\n[OK] Comparison dashboard: {comparison_html}")
    print(f"[OK] Tuning CSV:           {csv_path}")
    print(f"[OK] Tuning summary:       {summary_path}")


if __name__ == "__main__":
    main()
