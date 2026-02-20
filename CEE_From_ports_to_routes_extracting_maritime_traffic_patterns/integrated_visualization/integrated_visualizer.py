"""
Integrated Visualizer — Overlays PLSN and NLSN layers on a single dashboard.
Supports multiple hyperparameter-tuned layers and embedded scientific analysis.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd


class IntegratedVisualizer:
    """
    Generates an integrated HTML dashboard with multiple PLSN and NLSN layers.
    Includes technical analysis and statistics tabs.
    """

    def __init__(
        self,
        output_file: str,
        sample_size: int = 500_000,
    ):
        self.output_file = output_file
        self.sample_size = sample_size
        self.logger = logging.getLogger(__name__)
        
        # State for layers
        self.plsn_layers: dict[str, dict[str, Any]] = {}
        self.nlsn_layers: dict[str, dict[str, Any]] = {}
        self.heatmap_points: list[list[float]] = []
        self.methodology_md: str = ""

    def add_plsn_layer(
        self,
        name: str,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        boundaries_geojson: dict,
    ):
        self.plsn_layers[name] = {
            "nodes": self._process_nodes(nodes_df, "stationary_points"),
            "edges": self._process_edges(edges_df, nodes_df),
            "boundaries": boundaries_geojson,
            "stats": {
                "n_ports": len(nodes_df),
                "n_edges": len(edges_df),
                "total_transitions": int(edges_df["transition_count"].sum()) if not edges_df.empty else 0,
                "avg_density": float(nodes_df["stationary_points"].mean()) if not nodes_df.empty else 0
            }
        }

    def add_nlsn_layer(
        self,
        name: str,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        boundaries_geojson: dict,
        feature_points_df: pd.DataFrame | None = None,
    ):
        self.nlsn_layers[name] = {
            "nodes": self._process_nodes(nodes_df, "feature_points"),
            "edges": self._process_edges(edges_df, nodes_df),
            "boundaries": boundaries_geojson,
            "trajectories": self._reconstruct_trajectories(feature_points_df) if feature_points_df is not None else [],
            "stats": {
                "n_waypoints": len(nodes_df),
                "n_edges": len(edges_df),
                "total_transitions": int(edges_df["transition_count"].sum()) if not edges_df.empty else 0,
                "feature_points": len(feature_points_df) if feature_points_df is not None else 0
            }
        }

    def set_methodology(self, md_content: str):
        self.methodology_md = md_content

    def set_heatmap_data(self, ais_df: pd.DataFrame):
        lat_col = next((c for c in ["LAT", "Latitude", "lat"] if c.upper() in [x.upper() for x in ais_df.columns]), "LAT")
        lon_col = next((c for c in ["LON", "Longitude", "lon"] if c.upper() in [x.upper() for x in ais_df.columns]), "LON")
        
        actual_lat = next((c for c in ais_df.columns if c.upper() == lat_col.upper()), None)
        actual_lon = next((c for c in ais_df.columns if c.upper() == lon_col.upper()), None)

        if not actual_lat or not actual_lon:
            return

        work = ais_df[[actual_lat, actual_lon]].dropna()
        if len(work) > self.sample_size:
            work = work.sample(n=self.sample_size, random_state=42)
        self.heatmap_points = work[[actual_lat, actual_lon]].values.tolist()

    def _process_nodes(self, df: pd.DataFrame, point_col_hint: str) -> list[dict]:
        if df.empty: return []
        point_col = next((c for c in df.columns if c.upper() == point_col_hint.upper()), None) or df.columns[0]
        nodes_sorted = df.sort_values(point_col, ascending=False).reset_index(drop=True)
        nodes = []
        for rank, row in enumerate(nodes_sorted.itertuples(index=False)):
            nodes.append({
                "port_id": int(getattr(row, "port_id")),
                "lat": float(getattr(row, "lat")),
                "lon": float(getattr(row, "lon")),
                "points": int(getattr(row, point_col, 0)),
                "rank": rank + 1
            })
        return nodes

    def _process_edges(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> list[dict]:
        if edges_df.empty or nodes_df.empty: return []
        node_lookup = {int(row.port_id): (float(row.lat), float(row.lon)) for row in nodes_df.itertuples()}
        edges = []
        src_col = "source_port" if "source_port" in edges_df.columns else "source_id"
        dst_col = "target_port" if "target_port" in edges_df.columns else "target_id"
        for row in edges_df.itertuples():
            src, dst = int(getattr(row, src_col)), int(getattr(row, dst_col))
            if src in node_lookup and dst in node_lookup:
                edges.append({
                    "src": src, "dst": dst,
                    "src_lat": node_lookup[src][0], "src_lon": node_lookup[src][1],
                    "dst_lat": node_lookup[dst][0], "dst_lon": node_lookup[dst][1],
                    "vessels": int(getattr(row, "unique_vessels", 0)),
                    "transitions": int(getattr(row, "transition_count", 0))
                })
        return edges

    def _reconstruct_trajectories(self, df: pd.DataFrame, max_segments: int = 1500) -> list[list[list[float]]]:
        if df.empty: return []
        time_col = next((c for c in df.columns if c.upper() in ["BASEDATETIME", "TIMESTAMP", "TIME"]), None)
        lat_col = next((c for c in df.columns if c.upper() == "LAT"), None)
        lon_col = next((c for c in df.columns if c.upper() == "LON"), None)
        traj_col = next((c for c in df.columns if c.upper() == "TRAJECTORY_ID"), "MMSI")
        if not all([time_col, lat_col, lon_col]): return []
        df = df.sort_values([traj_col, time_col])
        unique_trajs = df[traj_col].unique()
        if len(unique_trajs) > max_segments:
            import numpy as np
            np.random.seed(42)
            selected = np.random.choice(unique_trajs, max_segments, replace=False)
            df_plot = df[df[traj_col].isin(selected)]
        else: df_plot = df
        trajectories = []
        for _, group in df_plot.groupby(traj_col):
            if len(group) >= 2: trajectories.append(group[[lat_col, lon_col]].values.tolist())
        return trajectories

    def generate_dashboard(self) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        center_lat, center_lon = 15.0, 85.0
        
        # Prepare Stats Data for Charts
        labels = list(self.plsn_layers.keys())
        plsn_nodes = [self.plsn_layers[k]["stats"]["n_ports"] for k in labels]
        plsn_edges = [self.plsn_layers[k]["stats"]["n_edges"] for k in labels]
        
        n_labels = list(self.nlsn_layers.keys())
        nlsn_nodes = [self.nlsn_layers[k]["stats"]["n_waypoints"] for k in n_labels]
        nlsn_trans = [self.nlsn_layers[k]["stats"]["total_transitions"] for k in n_labels]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Maritime Research Portal - Multiscale Integration</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; height: 100vh; background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; }}
        header {{ background: #1e293b; padding: 0 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; height: 60px; z-index: 1001; }}
        .tabs {{ display: flex; gap: 4px; height: 100%; align-items: center; }}
        .tab-btn {{ padding: 8px 16px; background: transparent; border: none; color: #94a3b8; cursor: pointer; border-radius: 6px; font-weight: 500; transition: all 0.2s; }}
        .tab-btn.active {{ background: #334155; color: #38bdf8; }}
        .tab-content {{ flex: 1; display: none; overflow: hidden; position: relative; }}
        .tab-content.active {{ display: flex; }}
        
        #map {{ width: 100%; height: 100%; }}
        #analysis-view {{ overflow-y: auto; padding: 40px; width: 100%; box-sizing: border-box; display: flex; justify-content: center; }}
        .md-container {{ max-width: 900px; line-height: 1.7; color: #cbd5e1; }}
        .md-container h1, .md-container h2 {{ color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-top: 40px; }}
        
        #stats-view {{ overflow-y: auto; padding: 40px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; box-sizing: border-box; }}
        .chart-box {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; min-height: 400px; }}
        
        .layer-ctrl-custom {{ position: absolute; bottom: 20px; left: 20px; z-index: 1000; background: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 8px; max-height: 300px; overflow-y: auto; }}
        .stat-summary {{ font-size: 0.8rem; color: #64748b; margin-top: 4px; }}
    </style>
</head>
<body>
    <header>
        <div style="display: flex; align-items:center; gap: 15px;">
            <h1 style="font-size: 1.2rem; margin:0;">Scientific Research Portal</h1>
            <span style="color: #475569;">|</span>
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('map-view')">Interactive Map</button>
                <button class="tab-btn" onclick="showTab('analysis-view')">Methodology & Analysis</button>
                <button class="tab-btn" onclick="showTab('stats-view')">Results & Metrics</button>
            </div>
        </div>
        <div style="font-size: 0.8rem; color: #64748b;">Generated: {datetime.now().strftime('%Y-%m-%d')}</div>
    </header>

    <div id="map-view" class="tab-content active">
        <div id="map"></div>
    </div>

    <div id="analysis-view" class="tab-content">
        <div class="md-container" id="methodology-content"></div>
    </div>

    <div id="stats-view" class="tab-content">
        <div class="chart-box"><canvas id="plsnChart"></canvas></div>
        <div class="chart-box"><canvas id="nlsnChart"></canvas></div>
        <div class="chart-box"><canvas id="edgeChart"></canvas></div>
        <div class="chart-box" style="padding: 24px;">
            <h3 style="margin-top: 0; color: #38bdf8;">Data Summary</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">
                <div style="background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">Total AIS Samples</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8;">{len(self.heatmap_points):,}</div>
                </div>
                <div style="background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">Hyperparameter Trials</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #10b981;">{len(labels) + len(n_labels)}</div>
                </div>
            </div>
            <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 20px; line-height: 1.6;">
                The charts visualize the scaling behavior of the maritime network extraction. 
                Higher density thresholds (lower <i>r</i>) and grid granularity (high <i>K</i>) result in more fragmented networks but higher precision in port localization.
            </p>
        </div>
    </div>

    <script>
        // 1. Tab Logic
        function showTab(id) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.currentTarget.classList.add('active');
            if (id === 'map-view') map.invalidateSize();
        }}

        // 2. Markdown Rendering
        document.getElementById('methodology-content').innerHTML = marked.parse({json.dumps(self.methodology_md)});

        // 3. Map Initialization
        const map = L.map('map').setView([{center_lat}, {center_lon}], 4);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
        }}).addTo(map);

        const heatmapData = {json.dumps(self.heatmap_points)};
        const plsnSets = {json.dumps(self.plsn_layers)};
        const nlsnSets = {json.dumps(self.nlsn_layers)};

        const heatLayer = L.heatLayer(heatmapData, {{radius: 8, blur: 12, minOpacity: 0.2}}).addTo(map);
        const overlays = {{ "Base: AIS Heatmap": heatLayer }};
        
        // PLSN Layers
        Object.keys(plsnSets).forEach(name => {{
            const data = plsnSets[name];
            const group = L.layerGroup();
            L.geoJSON(data.boundaries, {{ style: {{ color: '#dc2626', weight: 2, fillOpacity: 0.1 }} }}).addTo(group);
            data.nodes.forEach(n => L.circleMarker([n.lat, n.lon], {{ radius: 5, fillColor: '#3b82f6', fillOpacity: 0.8, weight: 1 }}).bindPopup(`Port ${{n.port_id}}`).addTo(group));
            overlays[`PLSN scale: ${{name}}`] = group;
            if (Object.keys(plsnSets).indexOf(name) === 0) group.addTo(map);
        }});

        // NLSN Layers
        Object.keys(nlsnSets).forEach(name => {{
            const data = nlsnSets[name];
            const group = L.layerGroup();
            L.geoJSON(data.boundaries, {{ style: {{ color: '#10b981', weight: 1.5, fillOpacity: 0.1 }} }}).addTo(group);
            data.trajectories.forEach(t => L.polyline(t, {{ color: '#10b981', weight: 0.8, opacity: 0.4 }}).addTo(group));
            overlays[`NLSN scale: ${{name}}`] = group;
        }});

        L.control.layers(null, overlays, {{collapsed: true}}).addTo(map);

        // 4. Charts Logic
        const ctxP = document.getElementById('plsnChart').getContext('2d');
        new Chart(ctxP, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{ label: 'Port Count', data: {json.dumps(plsn_nodes)}, backgroundColor: '#3b82f6' }}]
            }},
            options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'PLSN Discovery (Ports per Config)', color: '#fff' }} }} }}
        }});

        const ctxN = document.getElementById('nlsnChart').getContext('2d');
        new Chart(ctxN, {{
            type: 'line',
            data: {{
                labels: {json.dumps(n_labels)},
                datasets: [{{ label: 'Waypoint Hubs', data: {json.dumps(nlsn_nodes)}, borderColor: '#10b981', tension: 0.4 }}]
            }},
            options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'NLSN Resolution (Waypoints per Gamma)', color: '#fff' }} }} }}
        }});

        const ctxE = document.getElementById('edgeChart').getContext('2d');
        new Chart(ctxE, {{
            type: 'scatter',
            data: {{
                datasets: [{{ label: 'Edges', data: {json.dumps([{"x": n, "y": e} for n, e in zip(plsn_nodes, plsn_edges)])}, backgroundColor: '#dc2626' }}]
            }},
            options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Network Complexity (Nodes vs Edges)', color: '#fff' }} }} }}
        }});
    </script>
</body>
</html>
"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info("Integrated Portal saved to %s", self.output_file)
