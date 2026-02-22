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
        self.rlsn_layers: dict[str, dict[str, Any]] = {}
        self.heatmap_points: list[list[float]] = []
        self.methodology_md: str = ""

    def add_plsn_layer(
        self,
        name: str,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        boundaries_geojson: dict,
        feature_points_df: pd.DataFrame | None = None,
        batch: str = "Default",
    ):
        self.plsn_layers[name] = {
            "nodes": self._process_nodes(nodes_df, "stationary_points"),
            "edges": self._process_edges(edges_df, nodes_df),
            "boundaries": boundaries_geojson,
            "trajectories": self._reconstruct_trajectories(feature_points_df) if feature_points_df is not None else [],
            "batch": batch,
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
        batch: str = "Default",
    ):
        self.nlsn_layers[name] = {
            "nodes": self._process_nodes(nodes_df, "feature_points"),
            "edges": self._process_edges(edges_df, nodes_df),
            "boundaries": boundaries_geojson,
            "trajectories": self._reconstruct_trajectories(feature_points_df) if feature_points_df is not None else [],
            "batch": batch,
            "stats": {
                "n_waypoints": len(nodes_df),
                "n_edges": len(edges_df),
                "total_transitions": int(edges_df["transition_count"].sum()) if not edges_df.empty else 0,
                "feature_points": len(feature_points_df) if feature_points_df is not None else 0
            }
        }

    def add_rlsn_layer(
        self,
        name: str,
        routes_geojson: dict,
        boundaries_geojson: dict,
        batch: str = "Default",
    ):
        self.rlsn_layers[name] = {
            "routes": routes_geojson,
            "boundaries": boundaries_geojson,
            "batch": batch,
            "stats": {
                "n_routes": len(routes_geojson.get("features", [])),
                "n_boundary_polys": len(boundaries_geojson.get("features", []))
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
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Maritime Analytics - Port Discovery Workstation</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; height: 100vh; background: #020617; color: #f8fafc; display: flex; flex-direction: column; overflow: hidden; }}
        header {{ background: #0f172a; padding: 0 16px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; height: 40px; z-index: 1001; flex-shrink: 0; }}
        .tabs {{ display: flex; gap: 2px; height: 100%; align-items: center; }}
        .tab-btn {{ padding: 4px 10px; background: transparent; border: none; color: #64748b; cursor: pointer; border-radius: 4px; font-weight: 600; transition: all 0.2s; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .tab-btn.active {{ background: #1e293b; color: #38bdf8; }}
        .tab-content {{ flex: 1; display: none; overflow: hidden; position: relative; }}
        .tab-content.active {{ display: flex; }}
        .map-container {{ flex: 1; height: 100%; position: relative; }}
        .full-map {{ width: 100%; height: 100%; }}
        #toggle-panel {{ position: absolute; bottom: 20px; right: 20px; z-index: 2001; background: #38bdf8; color: #020617; border: none; border-radius: 50%; width: 44px; height: 44px; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; }}
        #details-panel {{ width: 280px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(20px); border-left: 1px solid rgba(255,255,255,0.05); padding: 16px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; transform: translateX(100%); transition: transform 0.4s; position: absolute; right: 0; top: 0; bottom: 0; z-index: 2000; }}
        #details-panel.open {{ transform: translateX(0); }}
        .panel-close {{ position: absolute; top: 12px; right: 12px; background: none; border: none; color: #64748b; cursor: pointer; font-size: 1.2rem; }}
        .detail-card {{ background: rgba(2, 6, 23, 0.5); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 10px; }}
        .detail-label {{ font-size: 0.6rem; color: #475569; text-transform: uppercase; font-weight: 800; }}
        .detail-value {{ font-size: 0.85rem; color: #f8fafc; font-weight: 600; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }}
    </style>
</head>
<body>
    <header>
        <div style="display: flex; align-items:center; gap: 12px;">
            <h1 style="font-size: 0.9rem; margin:0; color: #38bdf8; font-weight: 800;">PORT ANALYTICS WORKSTATION</h1>
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('macro-plsn')">Macro (PLSN)</button>
                <button class="tab-btn" onclick="showTab('meso-nlsn')">Meso (NLSN)</button>
                <button class="tab-btn" onclick="showTab('route-rlsn')">Route (RLSN)</button>
                <button class="tab-btn" onclick="showTab('integrated-ms')">Integration</button>
            </div>
        </div>
    </header>

    <button id="toggle-panel" onclick="togglePanel()" title="Toggle Profile">
        <svg style="width: 24px; height: 24px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
    </button>

    <aside id="details-panel">
        <button class="panel-close" onclick="closePanel()">×</button>
        <h2 id="panel-title" style="color: #f8fafc; margin: 0; font-size: 1rem;">Research Profile</h2>
        <div id="panel-content" style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;"></div>
    </aside>

    <div id="macro-plsn" class="tab-content active map-container"><div id="map-plsn" class="full-map"></div></div>
    <div id="meso-nlsn" class="tab-content map-container"><div id="map-nlsn" class="full-map"></div></div>
    <div id="route-rlsn" class="tab-content map-container"><div id="map-rlsn" class="full-map"></div></div>
    <div id="integrated-ms" class="tab-content map-container"><div id="map-integrated" class="full-map"></div></div>

    <script>
        const maps = {{ }};
        function showTab(id) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            if (window.event) window.event.currentTarget.classList.add('active');
            Object.values(maps).forEach(m => m.invalidateSize());
        }}
        function togglePanel() {{ document.getElementById('details-panel').classList.toggle('open'); }}
        function closePanel() {{ document.getElementById('details-panel').classList.remove('open'); }}

        const plsnSets = {json.dumps(self.plsn_layers)};
        const nlsnSets = {json.dumps(self.nlsn_layers)};
        const rlsnSets = {json.dumps(self.rlsn_layers)};

        function initMap(id) {{
            const m = L.map(id).setView([{center_lat}, {center_lon}], 4);
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; OpenStreetMap'
            }}).addTo(m);
            return m;
        }}

        maps['plsn'] = initMap('map-plsn');
        maps['nlsn'] = initMap('map-nlsn');
        maps['rlsn'] = initMap('map-rlsn');
        maps['integrated'] = initMap('map-integrated');

        let activeSelection = null;
        function updateInfoPanel(type, name, data, layer) {{
            if (activeSelection) activeSelection.setStyle({{ color: activeSelection._originalColor || '#fff', weight: activeSelection._originalWeight || 1 }});
            activeSelection = layer;
            layer._originalColor = layer.options.color;
            layer._originalWeight = layer.options.weight;
            layer.setStyle({{ color: '#38bdf8', weight: 4 }});

            document.getElementById('panel-content').innerHTML = `
                <div class="detail-card"><div class="detail-label">Object Type</div><div class="detail-value">${{type}}</div></div>
                <div class="detail-card"><div class="detail-label">Batch Config</div><div class="detail-value">${{data.batch || "Global"}}</div></div>
                <div class="detail-card"><div class="detail-label">Trial Metadata</div><div class="detail-value">${{name}}</div></div>
                <div class="detail-card"><div class="detail-label">Coordinates</div><div class="detail-value">${{data.lat.toFixed(4)}}, ${{data.lon.toFixed(4)}}</div></div>
            `;
            document.getElementById('details-panel').classList.add('open');
        }}

        const controls = {{
            plsn: L.control.layers(null, {{}}, {{ collapsed: false }}).addTo(maps['plsn']),
            nlsn: L.control.layers(null, {{}}, {{ collapsed: false }}).addTo(maps['nlsn']),
            rlsn: L.control.layers(null, {{}}, {{ collapsed: false }}).addTo(maps['rlsn']),
            integrated: L.control.layers(null, {{}}, {{ collapsed: false }}).addTo(maps['integrated'])
        }};

        function addLayerSet(sets, typeLabel, mapKey, color, radius) {{
            Object.entries(sets).forEach(([name, data]) => {{
                const group = L.layerGroup();
                const bName = `[${{data.batch}}] ${{typeLabel}}: ${{name}}`;
                
                if (data.nodes) data.nodes.forEach(n => {{
                    const m = L.circleMarker([n.lat, n.lon], {{ radius: radius, fillColor: color, color: '#fff', weight: 0.5, fillOpacity: 0.8 }});
                    m.on('click', (e) => updateInfoPanel(typeLabel, name, {{...n, batch: data.batch}}, e.target));
                    m.addTo(group);
                }});
                if (data.edges) data.edges.forEach(e => L.polyline([[e.src_lat, e.src_lon], [e.dst_lat, e.dst_lon]], {{ color: color, weight: 1.2, opacity: 0.3 }}).addTo(group));
                if (data.routes) L.geoJSON(data.routes, {{ style: {{ color: color, weight: 2.5, opacity: 0.8 }} }}).addTo(group);
                if (data.boundaries) L.geoJSON(data.boundaries, {{ style: {{ color: color === '#3b82f6' ? '#ef4444' : '#10b981', weight: 1.5, fillOpacity: 0.1, dashArray: '4,4' }} }}).addTo(group);
                if (data.trajectories) data.trajectories.forEach(t => L.polyline(t, {{ color: '#f59e0b', weight: 0.8, opacity: 0.2 }}).addTo(group));

                controls[mapKey].addOverlay(group, bName);
                controls.integrated.addOverlay(group, bName);
                if (mapKey !== 'integrated') group.addTo(maps[mapKey]);
            }});
        }}

        addLayerSet(plsnSets, "Port", "plsn", "#3b82f6", 4);
        addLayerSet(nlsnSets, "Waypoint", "nlsn", "#10b981", 3);
        addLayerSet(rlsnSets, "Route", "rlsn", "#38bdf8", 2);
    </script>
</body>
</html>
"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info("Port Analytics Workstation saved to %s", self.output_file)
