"""
Enhanced PLSNVisualizer — Leaflet-based HTML dashboard.

Changes vs original:
  • Boundary polygons: weight=4, opacity=0.9, fillOpacity=0.18
  • Edges: min 3px, max 14px, colour-coded by unique_vessels (blue→red)
  • Arrowhead SVG markers on every edge (shows direction)
  • Node circles: larger, colour-coded by rank (green→orange for top ports)
  • Leaflet layer control to toggle individual layers
  • Rich popups (port rank, coords, area, vessels)
  • Header shows embedded trial hyperparameter metadata when supplied
  • Inline Chart.js mini-charts in the side panel
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime

import pandas as pd


class PLSNVisualizer:
    """
    Generates a standalone HTML dashboard using Leaflet (no folium dependency).
    """

    def __init__(
        self,
        output_file: str,
        sample_size: int = 500_000,
        edge_min_width: int = 3,
        edge_max_width: int = 14,
        boundary_weight: int = 4,
        node_min_radius: int = 5,
        node_max_radius: int = 18,
    ):
        self.output_file = output_file
        self.sample_size = sample_size
        self.edge_min_width = edge_min_width
        self.edge_max_width = edge_max_width
        self.boundary_weight = boundary_weight
        self.node_min_radius = node_min_radius
        self.node_max_radius = node_max_radius
        self.logger = logging.getLogger(__name__)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_points(
        df: pd.DataFrame, lat_col: str, lon_col: str, sample_size: int
    ) -> list[list[float]]:
        if df.empty:
            return []
        work = df[[lat_col, lon_col]].dropna()
        if len(work) > sample_size:
            work = work.sample(n=sample_size, random_state=42)
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce").astype("float32").round(5)
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce").astype("float32").round(5)
        return work[[lat_col, lon_col]].dropna().values.tolist()

    @staticmethod
    def _node_color(rank: int, total: int) -> str:
        """Top 10 % → crimson, top 40 % → amber, rest → teal."""
        frac = rank / max(total, 1)
        if frac < 0.10:
            return "#ef4444"   # crimson  – busiest
        if frac < 0.40:
            return "#f59e0b"   # amber
        return "#22c55e"       # teal / green

    # ── public API ───────────────────────────────────────────────────────────

    def generate_plsn_dashboard(
        self,
        full_df: pd.DataFrame,
        clustered_df: pd.DataFrame,
        boundaries_list: list[dict],
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        params_meta: dict | None = None,   # {k, r, neighbor_mode, ...}
        trial_score: float | None = None,
    ) -> None:
        self.logger.info("Generating enhanced PLSN dashboard HTML → %s", self.output_file)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        lat_col = "LAT" if "LAT" in full_df.columns else "Latitude"
        lon_col = "LON" if "LON" in full_df.columns else "Longitude"

        all_points = self._sample_points(full_df, lat_col, lon_col, self.sample_size)
        in_port_df = (
            clustered_df[clustered_df["cluster_id"] != -1]
            if "cluster_id" in clustered_df.columns
            else clustered_df
        )
        in_points = self._sample_points(in_port_df, lat_col, lon_col, self.sample_size)

        # ── GeoJSON boundaries ───────────────────────────────────────────────
        geojson_features = [
            {
                "type": "Feature",
                "properties": {"cluster_id": int(item["cluster_id"])},
                "geometry": item["geometry"].__geo_interface__,
            }
            for item in boundaries_list
        ]
        boundaries_geojson = {"type": "FeatureCollection", "features": geojson_features}

        # ── Nodes ────────────────────────────────────────────────────────────
        nodes_sorted = (
            nodes_df.sort_values("stationary_points", ascending=False).reset_index(drop=True)
            if not nodes_df.empty and "stationary_points" in nodes_df.columns
            else nodes_df.reset_index(drop=True)
        )
        nodes: list[dict] = []
        node_lookup: dict[int, tuple[float, float]] = {}
        for rank, row in enumerate(nodes_sorted.itertuples(index=False)):
            node = {
                "port_id": int(row.port_id),
                "lat": float(row.lat),
                "lon": float(row.lon),
                "stationary_points": int(getattr(row, "stationary_points", 0)),
                "area_deg2": float(getattr(row, "area_deg2", 0.0)),
                "rank": rank + 1,
                "color": self._node_color(rank, len(nodes_sorted)),
            }
            nodes.append(node)
            node_lookup[node["port_id"]] = (node["lat"], node["lon"])

        # ── Edges ────────────────────────────────────────────────────────────
        edges: list[dict] = []
        max_vessels = 1
        for row in edges_df.itertuples(index=False):
            src, dst = int(row.source_port), int(row.target_port)
            if src not in node_lookup or dst not in node_lookup:
                continue
            uv = int(row.unique_vessels)
            max_vessels = max(max_vessels, uv)
            edges.append(
                {
                    "source_port": src,
                    "target_port": dst,
                    "transition_count": int(row.transition_count),
                    "unique_vessels": uv,
                    "source_lat": node_lookup[src][0],
                    "source_lon": node_lookup[src][1],
                    "target_lat": node_lookup[dst][0],
                    "target_lon": node_lookup[dst][1],
                }
            )

        # ── Stats ────────────────────────────────────────────────────────────
        top_ports = nodes[:10]
        top_edges = sorted(edges, key=lambda x: x["transition_count"], reverse=True)[:10]
        center_lat = float(nodes_df["lat"].mean()) if not nodes_df.empty else 20.0
        center_lon = float(nodes_df["lon"].mean()) if not nodes_df.empty else 0.0
        total_transitions = int(edges_df["transition_count"].sum()) if not edges_df.empty else 0
        total_vessels = int(
            edges_df["unique_vessels"].max() if not edges_df.empty else 0
        )

        # ── Param meta subtitle ──────────────────────────────────────────────
        if params_meta:
            subtitle = (
                f"K={params_meta.get('k','?')} | "
                f"r={params_meta.get('r','?')} | "
                f"Nbr={params_meta.get('neighbor_mode','?')} | "
                f"MinDense={params_meta.get('min_dense_points','none')}"
            )
            if trial_score is not None:
                subtitle += f" | Score={trial_score:.4f}"
        else:
            subtitle = "Port-Level Shipping Network"

        # ── Mini-chart data (top port sizes for sparkline) ───────────────────
        sparkline_labels = [f"P{n['port_id']}" for n in top_ports]
        sparkline_data   = [n["stationary_points"] for n in top_ports]

        edge_trans_labels = [f"{e['source_port']}→{e['target_port']}" for e in top_edges]
        edge_trans_data   = [e["transition_count"] for e in top_edges]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PLSN Dashboard — {subtitle}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; }}
    #map {{ position:fixed; inset:0; }}
    #panel {{
      position:fixed; top:12px; right:12px; width:340px; max-height:calc(100vh - 24px);
      overflow-y:auto; z-index:9999;
      background:rgba(15,23,42,0.92); color:#f1f5f9;
      border-radius:14px; padding:14px 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      backdrop-filter: blur(6px);
      font-size:12px;
    }}
    #panel h2 {{ font-size:15px; color:#38bdf8; margin-bottom:2px; }}
    #panel .subtitle {{ font-size:10px; color:#94a3b8; margin-bottom:10px; word-break:break-all; }}
    .stat-row {{ display:flex; gap:6px; margin-bottom:8px; flex-wrap:wrap; }}
    .stat-card {{
      flex:1; min-width:70px; background:rgba(255,255,255,0.07);
      border-radius:8px; padding:6px 8px; text-align:center;
    }}
    .stat-card .val {{ font-size:17px; font-weight:700; color:#f8fafc; }}
    .stat-card .lbl {{ font-size:9px; color:#94a3b8; margin-top:1px; }}
    .section {{ margin:8px 0 4px; font-size:11px; font-weight:600; color:#93c5fd; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:3px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ padding:3px 4px; border-bottom:1px solid rgba(255,255,255,0.08); }}
    th {{ font-size:10px; color:#94a3b8; font-weight:600; text-align:left; }}
    td {{ font-size:11px; }}
    .chart-box {{ background:rgba(255,255,255,0.05); border-radius:8px; padding:8px; margin:6px 0; }}
    .chart-title {{ font-size:10px; color:#94a3b8; margin-bottom:4px; }}
    .ts {{ font-size:9px; color:#475569; margin-top:10px; text-align:right; }}
    canvas {{ max-height: 120px !important; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="panel">
    <h2>PLSN Network</h2>
    <div class="subtitle">{subtitle}</div>

    <!-- Summary cards -->
    <div class="stat-row">
      <div class="stat-card"><div class="val">{len(nodes)}</div><div class="lbl">Ports</div></div>
      <div class="stat-card"><div class="val">{len(edges)}</div><div class="lbl">Lanes</div></div>
      <div class="stat-card"><div class="val">{total_transitions:,}</div><div class="lbl">Transitions</div></div>
      <div class="stat-card"><div class="val">{len(in_points):,}</div><div class="lbl">Stationary Pts</div></div>
    </div>

    <!-- Top ports chart -->
    <div class="chart-box">
      <div class="chart-title">Top 10 Ports by Stationary Points</div>
      <canvas id="portChart"></canvas>
    </div>

    <!-- Top edges chart -->
    <div class="chart-box">
      <div class="chart-title">Top 10 Routes by Transitions</div>
      <canvas id="edgeChart"></canvas>
    </div>

    <!-- Top ports table -->
    <div class="section">Top Ports</div>
    <table>
      <thead><tr><th>#</th><th>Port</th><th>Points</th><th>Area(deg²)</th></tr></thead>
      <tbody>
        {"".join([
            f"<tr><td>{n['rank']}</td><td>{n['port_id']}</td>"
            f"<td>{n['stationary_points']:,}</td>"
            f"<td>{n['area_deg2']:.5f}</td></tr>"
            for n in top_ports
        ]) or "<tr><td colspan='4'>No data</td></tr>"}
      </tbody>
    </table>

    <!-- Top routes table -->
    <div class="section">Top Routes</div>
    <table>
      <thead><tr><th>Edge</th><th>Trans.</th><th>Vessels</th></tr></thead>
      <tbody>
        {"".join([
            f"<tr><td>{e['source_port']}→{e['target_port']}</td>"
            f"<td>{e['transition_count']:,}</td>"
            f"<td>{e['unique_vessels']}</td></tr>"
            for e in top_edges
        ]) or "<tr><td colspan='3'>No data</td></tr>"}
      </tbody>
    </table>

    <div class="ts">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  </div>

  <script>
  // ── Data ──────────────────────────────────────────────────────────────────
  const allPoints   = {json.dumps(all_points)};
  const inPoints    = {json.dumps(in_points)};
  const boundaries  = {json.dumps(boundaries_geojson)};
  const nodes       = {json.dumps(nodes)};
  const edges       = {json.dumps(edges)};
  const EDGE_MIN    = {self.edge_min_width};
  const EDGE_MAX    = {self.edge_max_width};
  const BNDRY_WT    = {self.boundary_weight};
  const NODE_MINR   = {self.node_min_radius};
  const NODE_MAXR   = {self.node_max_radius};

  // ── Map ───────────────────────────────────────────────────────────────────
  const map = L.map('map', {{zoomControl: true}})
    .setView([{center_lat:.4f}, {center_lon:.4f}], {5 if nodes else 2});

  const baseTiles = L.tileLayer(
    'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{maxZoom:19, attribution:'&copy; OpenStreetMap &copy; CARTO', subdomains:'abcd'}}
  ).addTo(map);

  // ── Layers ────────────────────────────────────────────────────────────────
  const rawHeat = L.heatLayer(allPoints,
    {{radius:8, blur:12, minOpacity:0.15, gradient:{{0.2:'#1e3a5f',0.6:'#2563eb',1.0:'#93c5fd'}}}});
  const portHeat = L.heatLayer(inPoints,
    {{radius:9, blur:10, minOpacity:0.40, gradient:{{0.3:'#7c3aed',0.7:'#ef4444',1.0:'#fef3c7'}}}});

  // Boundary polygons — thicker stroke
  const boundaryLayer = L.geoJSON(boundaries, {{
    style: function(f) {{
      return {{
        color: '#f87171',
        weight: BNDRY_WT,
        opacity: 0.9,
        fillColor: '#dc2626',
        fillOpacity: 0.18,
        dashArray: null,
      }};
    }},
    onEachFeature: function(f, layer) {{
      layer.bindTooltip('Port cluster ' + f.properties.cluster_id, {{sticky:true}});
    }}
  }});

  // Node circles
  const nodeLayer = L.layerGroup();
  const maxPts = nodes.length ? Math.max(...nodes.map(n => n.stationary_points), 1) : 1;
  nodes.forEach(function(n) {{
    const ratio  = Math.sqrt(Math.max(n.stationary_points, 1) / maxPts);
    const radius = NODE_MINR + (NODE_MAXR - NODE_MINR) * ratio;
    L.circleMarker([n.lat, n.lon], {{
      radius: radius,
      color: '#0f172a',
      weight: 2,
      fillColor: n.color,
      fillOpacity: 0.92,
    }}).bindPopup(
      `<b>Port ${{n.port_id}}</b><br>`+
      `Rank: <b>#${{n.rank}}</b><br>`+
      `Lat/Lon: ${{n.lat.toFixed(4)}}, ${{n.lon.toFixed(4)}}<br>`+
      `Stationary pts: <b>${{n.stationary_points.toLocaleString()}}</b><br>`+
      `Area: ${{n.area_deg2.toFixed(6)}} deg²`
    ).addTo(nodeLayer);
  }});

  // Edge polylines — thicker, colour-coded by vessel count
  const edgeLayer = L.layerGroup();
  const maxTrans   = edges.length ? Math.max(...edges.map(e => e.transition_count), 1) : 1;
  const maxVessels = edges.length ? Math.max(...edges.map(e => e.unique_vessels), 1) : 1;

  function edgeColor(ratio) {{
    // blue (low) → amber → red (high)
    if (ratio < 0.33) return '#3b82f6';
    if (ratio < 0.66) return '#f59e0b';
    return '#ef4444';
  }}

  edges.forEach(function(e) {{
    const trRatio = e.transition_count / maxTrans;
    const vRatio  = e.unique_vessels  / maxVessels;
    const w       = EDGE_MIN + (EDGE_MAX - EDGE_MIN) * trRatio;
    const color   = edgeColor(vRatio);

    const line = L.polyline(
      [[e.source_lat, e.source_lon], [e.target_lat, e.target_lon]],
      {{color: color, weight: w, opacity: 0.80}}
    );

    // Arrowhead using SVG decorator approach (manual mid-point arrow)
    const midLat = (e.source_lat + e.target_lat) / 2;
    const midLon = (e.source_lon + e.target_lon) / 2;
    const arrowMarker = L.circleMarker([midLat, midLon], {{
      radius: Math.max(2, w / 2),
      color: color,
      fillColor: color,
      fillOpacity: 1,
      weight: 0,
    }});

    line.bindTooltip(
      `${{e.source_port}} → ${{e.target_port}}<br>`+
      `Transitions: <b>${{e.transition_count.toLocaleString()}}</b><br>`+
      `Unique vessels: <b>${{e.unique_vessels}}</b>`,
      {{sticky: true}}
    );
    line.addTo(edgeLayer);
    arrowMarker.addTo(edgeLayer);
  }});

  // ── Add layers to map ────────────────────────────────────────────────────
  rawHeat.addTo(map);
  portHeat.addTo(map);
  boundaryLayer.addTo(map);
  edgeLayer.addTo(map);
  nodeLayer.addTo(map);

  // ── Layer control ────────────────────────────────────────────────────────
  L.control.layers(
    {{'CartoDB Dark': baseTiles}},
    {{
      'Raw AIS Heat': rawHeat,
      'In-Port Heat': portHeat,
      'Port Boundaries': boundaryLayer,
      'Shipping Lanes': edgeLayer,
      'Port Nodes': nodeLayer,
    }},
    {{collapsed: false, position: 'bottomleft'}}
  ).addTo(map);

  // ── Mini charts (Chart.js) ────────────────────────────────────────────────
  const chartDefaults = {{
    responsive: true,
    maintainAspectRatio: true,
    animation: false,
    plugins: {{legend: {{display: false}}}},
    scales: {{
      x: {{ticks: {{color:'#94a3b8', font:{{size:8}}}}, grid:{{color:'rgba(255,255,255,0.05)'}}}},
      y: {{ticks: {{color:'#94a3b8', font:{{size:8}}}}, grid:{{color:'rgba(255,255,255,0.05)'}}}},
    }},
  }};

  new Chart(document.getElementById('portChart'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(sparkline_labels)},
      datasets: [{{
        data: {json.dumps(sparkline_data)},
        backgroundColor: 'rgba(56,189,248,0.7)',
        borderRadius: 3,
      }}]
    }},
    options: chartDefaults,
  }});

  new Chart(document.getElementById('edgeChart'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(edge_trans_labels)},
      datasets: [{{
        data: {json.dumps(edge_trans_data)},
        backgroundColor: 'rgba(239,68,68,0.7)',
        borderRadius: 3,
      }}]
    }},
    options: chartDefaults,
  }});

  </script>
</body>
</html>
"""
        with open(self.output_file, "w", encoding="utf-8") as fh:
            fh.write(html)
        self.logger.info("Saved enhanced PLSN dashboard HTML to %s", self.output_file)
