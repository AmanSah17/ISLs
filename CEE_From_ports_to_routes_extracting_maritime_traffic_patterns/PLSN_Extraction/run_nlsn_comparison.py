"""
run_nlsn_comparison.py — Enhanced NLSN gamma sweep comparison dashboard.

Reads results/nlsn/nlsn_gamma_sweep_results.csv (or a provided path) and
generates an interactive HTML comparison dashboard with:
  - Summary cards (trials, best gamma, best LD, best score)
  - Chart.js charts:
      * Score vs Gamma (line, per w1/w2 combo)
      * Dr vs Dl scatter (points per trial, colored by score)
      * Nodes vs Gamma (line, w1=w2=1.0 subset)
      * w1 influence / w2 influence bar charts
  - Sortable all-trials table with color-coded score cells
  - CSV download button
  - Iframe map previewer for selected trial

Usage
─────
  python run_nlsn_comparison.py                          # auto-finds latest CSV
  python run_nlsn_comparison.py --results path/to.csv   # explicit
  python run_nlsn_comparison.py --top-maps 5            # link top-N maps
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from PLSN_Extraction import config


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("NLSN-Comparison")


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _score_style(val: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = (val - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    r = int(220 - t * 120)
    g = int(100 + t * 140)
    b = int(80)
    return f"background:{hex(r)}{hex(g)[2:]}{hex(b)[2:]};color:#fff;" if False else \
           f"background:hsl({int(t*120)},65%,38%);color:#fff;font-weight:700;"


def _ld_style(val: float) -> str:
    t = max(0.0, min(1.0, val / 2.0))     # LD in [0,2] — normalise
    return f"background:hsl({int(t*120)},55%,40%);color:#fff;"


# ── Table row builder ──────────────────────────────────────────────────────────

def _table_rows(df: pd.DataFrame, score_lo: float, score_hi: float,
                output_dir: str) -> str:
    rows = []
    for _, r in df.iterrows():
        sc   = float(r["score"])
        sst  = _score_style(sc, score_lo, score_hi)
        ldst = _ld_style(float(r["ld_score"]))
        map_rel = os.path.relpath(str(r["map_html"]), output_dir).replace("\\", "/") \
                  if pd.notna(r.get("map_html", "")) and str(r.get("map_html", "")) else ""
        link = f"<a href='#' onclick=\"loadMap('{map_rel}');return false;\">Preview</a>" if map_rel else "-"
        rows.append(
            "<tr>"
            f"<td>{int(r['trial'])}</td>"
            f"<td>{float(r['gamma']):.5g}</td>"
            f"<td>{float(r['w1']):.1f}</td>"
            f"<td>{float(r['w2']):.1f}</td>"
            f"<td style='{ldst}'>{float(r['ld_score']):.4f}</td>"
            f"<td>{float(r['compression_rate_dr']):.4f}</td>"
            f"<td>{float(r['distance_similarity_dl']):.4f}</td>"
            f"<td>{float(r['cluster_coverage']):.4f}</td>"
            f"<td>{int(r['nodes'])}</td>"
            f"<td>{int(r['edges'])}</td>"
            f"<td>{int(r['transitions'])}</td>"
            f"<td>{float(r['runtime_sec']):.2f}s</td>"
            f"<td style='{sst}'>{sc:.4f}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    return "".join(rows)


# ── Chart.js data serialisers ──────────────────────────────────────────────────

def _score_vs_gamma_data(df: pd.DataFrame) -> str:
    """One dataset per (w1,w2) combo, x=gamma sorted."""
    combos = df[["w1", "w2"]].drop_duplicates().sort_values(["w1", "w2"])
    datasets = []
    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706",
        "#7c3aed", "#db2777", "#0891b2", "#65a30d",
    ]
    for ci, (_, row) in enumerate(combos.iterrows()):
        w1, w2 = float(row["w1"]), float(row["w2"])
        subset = df[(df["w1"] == w1) & (df["w2"] == w2)].sort_values("gamma")
        pts = [{"x": float(r["gamma"]), "y": float(r["score"])} for _, r in subset.iterrows()]
        c   = colors[ci % len(colors)]
        datasets.append({
            "label": f"w1={w1:.1f} w2={w2:.1f}",
            "data":  pts,
            "borderColor": c, "backgroundColor": c + "33",
            "tension": 0.3, "pointRadius": 4,
        })
    return json.dumps(datasets)


def _dr_dl_scatter(df: pd.DataFrame) -> str:
    """Scatter of Dr vs Dl, point colour encodes score."""
    pts = [
        {
            "x": float(r["compression_rate_dr"]),
            "y": float(r["distance_similarity_dl"]),
            "score": float(r["score"]),
            "gamma": float(r["gamma"]),
            "label": f"g={r['gamma']:.5g} w1={r['w1']:.1f} w2={r['w2']:.1f}",
        }
        for _, r in df.iterrows()
    ]
    return json.dumps(pts)


def _nodes_vs_gamma(df: pd.DataFrame) -> str:
    """Line: nodes vs gamma for default w1=w2=1 subset (or all if no such subset)."""
    sub = df[(df["w1"] == 1.0) & (df["w2"] == 1.0)].sort_values("gamma") if not df.empty else df
    if sub.empty:
        sub = df.sort_values("gamma")
    pts = [{"x": float(r["gamma"]), "y": int(r["nodes"])} for _, r in sub.iterrows()]
    return json.dumps(pts)


def _w_influence(df: pd.DataFrame, col: str) -> str:
    """Mean score per value of col (w1 or w2)."""
    grp = df.groupby(col)["score"].mean().reset_index().sort_values(col)
    labels = [f"{float(r[col]):.1f}" for _, r in grp.iterrows()]
    values = [float(r["score"]) for _, r in grp.iterrows()]
    return json.dumps({"labels": labels, "values": values})


# ── Main HTML builder ──────────────────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame, output_path: str, ts: str) -> str:
    output_dir  = os.path.dirname(output_path)
    score_lo    = float(df["score"].min()) if not df.empty else 0.0
    score_hi    = float(df["score"].max()) if not df.empty else 1.0
    best        = df.iloc[0] if not df.empty else None

    n_trials    = len(df)
    best_gamma  = f"{float(best['gamma']):.5g}" if best is not None else "N/A"
    best_w1     = f"{float(best['w1']):.1f}"    if best is not None else "N/A"
    best_w2     = f"{float(best['w2']):.1f}"    if best is not None else "N/A"
    best_score  = f"{float(best['score']):.4f}" if best is not None else "N/A"
    best_ld     = f"{float(best['ld_score']):.4f}" if best is not None else "N/A"
    best_nodes  = str(int(best["nodes"]))   if best is not None else "N/A"
    best_edges  = str(int(best["edges"]))   if best is not None else "N/A"

    default_map = ""
    if best is not None and pd.notna(best.get("map_html", "")):
        default_map = os.path.relpath(str(best["map_html"]), output_dir).replace("\\", "/")

    table_rows   = _table_rows(df, score_lo, score_hi, output_dir)
    ds_score_vs_gamma = _score_vs_gamma_data(df)
    ds_scatter   = _dr_dl_scatter(df)
    ds_nodes     = _nodes_vs_gamma(df)
    ds_w1        = _w_influence(df, "w1")
    ds_w2        = _w_influence(df, "w2")

    # CSV inline for download
    csv_data = df.drop(columns=["trial_dir", "feature_points_csv", "map_html"],
                       errors="ignore").to_csv(index=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>NLSN Gamma Sweep Comparison — {ts}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
    h1{{background:linear-gradient(135deg,#2563eb,#7c3aed);padding:18px 24px;font-size:1.4rem;letter-spacing:.5px}}
    h1 span{{font-size:.85rem;opacity:.75;margin-left:12px}}
    .container{{padding:16px 20px;display:flex;flex-direction:column;gap:16px}}
    /* Summary cards */
    .cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px;text-align:center}}
    .card .val{{font-size:1.55rem;font-weight:700;color:#60a5fa;margin-bottom:4px}}
    .card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px}}
    /* Charts */
    .charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
    .charts .chart-half{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
    @media(max-width:900px){{.charts{{grid-template-columns:1fr}}}}
    .chart-box{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px}}
    .chart-box h3{{font-size:.8rem;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}}
    canvas{{max-height:260px}}
    /* Table */
    .tbl-wrap{{background:#1e293b;border:1px solid #334155;border-radius:10px;overflow:hidden}}
    .tbl-top{{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #334155}}
    .tbl-top h3{{font-size:.9rem;color:#94a3b8}}
    .dl-btn{{background:#2563eb;color:#fff;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:.8rem}}
    .dl-btn:hover{{background:#1d4ed8}}
    .scroll{{overflow:auto;max-height:420px}}
    table{{width:100%;border-collapse:collapse;font-size:.78rem}}
    th,td{{padding:6px 8px;text-align:left;white-space:nowrap;border-bottom:1px solid #1e293b}}
    th{{background:#0f172a;color:#94a3b8;position:sticky;top:0;cursor:pointer;user-select:none}}
    th:hover{{color:#60a5fa}}
    tr:hover td{{background:#1a2842}}
    /* Map preview */
    .map-section{{background:#1e293b;border:1px solid #334155;border-radius:10px;overflow:hidden}}
    .map-section h3{{padding:10px 14px;font-size:.85rem;color:#94a3b8;border-bottom:1px solid #334155}}
    iframe#map-frame{{width:100%;height:680px;border:none;background:#fff}}
    /* Best params banner */
    .banner{{background:linear-gradient(135deg,#1d4ed8,#4f46e5);border-radius:10px;padding:14px 20px;display:flex;gap:24px;flex-wrap:wrap}}
    .bp{{text-align:center}}
    .bp .v{{font-size:1.2rem;font-weight:700;color:#fff}}
    .bp .l{{font-size:.7rem;color:#bfdbfe;text-transform:uppercase}}
  </style>
</head>
<body>
<h1>NLSN Gamma Hyperparameter Comparison <span>Generated {ts}</span></h1>
<div class="container">

  <!-- Best params banner -->
  <div class="banner">
    <div class="bp"><div class="v">{best_gamma}</div><div class="l">Best Gamma</div></div>
    <div class="bp"><div class="v">{best_w1}</div><div class="l">w1</div></div>
    <div class="bp"><div class="v">{best_w2}</div><div class="l">w2</div></div>
    <div class="bp"><div class="v">{best_score}</div><div class="l">Score</div></div>
    <div class="bp"><div class="v">{best_ld}</div><div class="l">LD Score</div></div>
    <div class="bp"><div class="v">{best_nodes}</div><div class="l">Nodes</div></div>
    <div class="bp"><div class="v">{best_edges}</div><div class="l">Edges</div></div>
  </div>

  <!-- Summary cards -->
  <div class="cards">
    <div class="card"><div class="val">{n_trials}</div><div class="lbl">Total Trials</div></div>
    <div class="card"><div class="val">{len(df['gamma'].unique()) if not df.empty else 0}</div><div class="lbl">Gamma Values</div></div>
    <div class="card"><div class="val">{len(df['w1'].unique()) if not df.empty else 0}</div><div class="lbl">w1 Values</div></div>
    <div class="card"><div class="val">{len(df['w2'].unique()) if not df.empty else 0}</div><div class="lbl">w2 Values</div></div>
    <div class="card"><div class="val">{best_score}</div><div class="lbl">Top Score</div></div>
    <div class="card"><div class="val">{best_ld}</div><div class="lbl">Top LD</div></div>
  </div>

  <!-- Charts row 1: Score vs Gamma + Dr/Dl scatter -->
  <div class="charts">
    <div class="chart-box">
      <h3>Score vs Gamma (per w1/w2 combo)</h3>
      <canvas id="scoreGammaChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Compression Rate (Dr) vs Fidelity (Dl)</h3>
      <canvas id="drDlChart"></canvas>
    </div>
  </div>

  <!-- Charts row 2: Nodes vs Gamma + w1/w2 influence -->
  <div class="charts">
    <div class="chart-box">
      <h3>Nodes vs Gamma (w1=w2=1.0)</h3>
      <canvas id="nodesGammaChart"></canvas>
    </div>
    <div class="chart-box chart-half" style="display:grid;grid-template-columns:1fr 1fr;gap:10px;background:transparent;border:none;padding:0">
      <div class="chart-box"><h3>Mean Score vs w1</h3><canvas id="w1Chart"></canvas></div>
      <div class="chart-box"><h3>Mean Score vs w2</h3><canvas id="w2Chart"></canvas></div>
    </div>
  </div>

  <!-- All-trials table -->
  <div class="tbl-wrap">
    <div class="tbl-top">
      <h3>All Trials ({n_trials})</h3>
      <button class="dl-btn" onclick="downloadCSV()">Download CSV</button>
    </div>
    <div class="scroll">
      <table id="results-table">
        <thead>
          <tr>
            <th onclick="sortTable(0)">#</th>
            <th onclick="sortTable(1)">Gamma</th>
            <th onclick="sortTable(2)">w1</th>
            <th onclick="sortTable(3)">w2</th>
            <th onclick="sortTable(4)">LD</th>
            <th onclick="sortTable(5)">Dr</th>
            <th onclick="sortTable(6)">Dl</th>
            <th onclick="sortTable(7)">Coverage</th>
            <th onclick="sortTable(8)">Nodes</th>
            <th onclick="sortTable(9)">Edges</th>
            <th onclick="sortTable(10)">Transitions</th>
            <th onclick="sortTable(11)">Runtime</th>
            <th onclick="sortTable(12)">Score</th>
            <th>Map</th>
          </tr>
        </thead>
        <tbody id="table-body">
{table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Map preview -->
  <div class="map-section">
    <h3 id="map-title">Best Trial Map Preview</h3>
    <iframe id="map-frame" src="{default_map}"></iframe>
  </div>

</div>

<script>
const CSV_DATA = {json.dumps(csv_data)};

// --- CSV download ---
function downloadCSV() {{
  const blob = new Blob([CSV_DATA], {{type:'text/csv'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'nlsn_sweep_results.csv'; a.click();
}}

// --- Map loader ---
function loadMap(rel) {{
  document.getElementById('map-frame').src = rel;
  document.getElementById('map-title').textContent = 'Map: ' + rel;
}}

// --- Sort table ---
let sortDir = {{}};
function sortTable(col) {{
  const tbody = document.getElementById('table-body');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const dir   = (sortDir[col] = !(sortDir[col]));
  rows.sort((a,b) => {{
    const av = a.cells[col].textContent.trim().replace('s','');
    const bv = b.cells[col].textContent.trim().replace('s','');
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an)&&!isNaN(bn)) return dir?(an-bn):(bn-an);
    return dir?av.localeCompare(bv):bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// --- Chart.js helpers ---
const CHART_DEF = {{responsive:true,animation:false,plugins:{{legend:{{labels:{{color:'#94a3b8',boxWidth:12,font:{{size:11}}}}}}}}}};

// 1. Score vs Gamma
const sgDatasets = {ds_score_vs_gamma};
new Chart(document.getElementById('scoreGammaChart'), {{
  type: 'line',
  data: {{datasets: sgDatasets}},
  options: {{
    ...CHART_DEF,
    scales: {{
      x: {{type:'logarithmic', title:{{display:true,text:'Gamma (log)',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}},
      y: {{title:{{display:true,text:'Score',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}}
    }},
    plugins: {{...CHART_DEF.plugins, tooltip:{{callbacks:{{label:(ctx)=>`${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(4)}}`}}}}}}
  }}
}});

// 2. Dr vs Dl scatter
const scPts = {ds_scatter};
const scores = scPts.map(p=>p.score);
const minS = Math.min(...scores), maxS = Math.max(...scores);
const scatterDs = scPts.map(p => {{
  const t = maxS>minS?(p.score-minS)/(maxS-minS):0.5;
  const h = Math.floor(t*120);
  return {{x:p.x, y:p.y, label:p.label, color:`hsl(${{h}},65%,55%)`}};
}});
new Chart(document.getElementById('drDlChart'), {{
  type:'scatter',
  data:{{datasets:[{{
    label:'Trials',
    data: scatterDs,
    backgroundColor: scatterDs.map(p=>p.color),
    pointRadius:5
  }}]}},
  options:{{
    ...CHART_DEF,
    scales:{{
      x:{{title:{{display:true,text:'Dr (Compression Rate)',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}},
      y:{{title:{{display:true,text:'Dl (Distance Fidelity)',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}}
    }},
    plugins:{{...CHART_DEF.plugins, tooltip:{{callbacks:{{label:(ctx)=>ctx.raw.label}}}}}}
  }}
}});

// 3. Nodes vs Gamma
const ngPts = {ds_nodes};
new Chart(document.getElementById('nodesGammaChart'), {{
  type:'line',
  data:{{datasets:[{{
    label:'Nodes (w1=w2=1)', data:ngPts,
    borderColor:'#34d399', backgroundColor:'#34d39933', tension:0.3, pointRadius:5
  }}]}},
  options:{{
    ...CHART_DEF,
    scales:{{
      x:{{type:'logarithmic',title:{{display:true,text:'Gamma',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}},
      y:{{title:{{display:true,text:'Nodes',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}}
    }}
  }}
}});

// 4. w1 influence
const w1d = {ds_w1};
new Chart(document.getElementById('w1Chart'), {{
  type:'bar',
  data:{{labels:w1d.labels, datasets:[{{label:'Mean Score',data:w1d.values,backgroundColor:'#2563eb99',borderColor:'#2563eb',borderWidth:1}}]}},
  options:{{...CHART_DEF,scales:{{x:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}},y:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}}}}}}
}});

// 5. w2 influence
const w2d = {ds_w2};
new Chart(document.getElementById('w2Chart'), {{
  type:'bar',
  data:{{labels:w2d.labels, datasets:[{{label:'Mean Score',data:w2d.values,backgroundColor:'#7c3aed99',borderColor:'#7c3aed',borderWidth:1}}]}},
  options:{{...CHART_DEF,scales:{{x:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}},y:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e3a5f'}}}}}}}}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build enhanced NLSN comparison dashboard.")
    parser.add_argument("--results",  type=str, default=None,
                        help="Path to nlsn_gamma_sweep_results.csv (auto-detected if omitted).")
    parser.add_argument("--nlsn-dir", type=str, default=config.NLSN_DIR)
    parser.add_argument("--top-maps", type=int, default=config.NLSN_TOP_MAPS)
    args   = parser.parse_args()
    logger = setup_logging()

    # Auto-detect latest results CSV
    if args.results:
        csv_path = args.results
    else:
        pattern  = os.path.join(args.nlsn_dir, "nlsn_gamma_sweep_results.csv")
        csvs     = sorted(glob.glob(pattern), reverse=True)
        if not csvs:
            logger.error("No nlsn_gamma_sweep_results.csv found in %s. Run run_nlsn_full.py first.", args.nlsn_dir)
            sys.exit(1)
        csv_path = csvs[0]

    logger.info("Loading results from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("  %d rows loaded.", len(df))

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(args.nlsn_dir, f"nlsn_comparison_{ts}.html")
    logger.info("Building dashboard ...")
    build_dashboard(df, html_path, ts)
    logger.info("Dashboard saved to: %s", html_path)
    print(f"\n[OK] NLSN comparison dashboard: {html_path}")


if __name__ == "__main__":
    main()
