import json
import logging
import os
import pandas as pd
from typing import Any, Optional

class ModularVisualizer:
    """
    Modular version of the Integrated Visualizer for the AIS Processor.
    Generates a Leaflet-based HTML dashboard.
    """
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.logger = logging.getLogger("ModularVisualizer")
        self.plsn_layers = {}
        self.nlsn_layers = {}
        self.rlsn_layers = {}
        self.heatmap_points = []

    def set_heatmap_data(self, ais_df: pd.DataFrame, max_points: int = 500000):
        """Sets data for the global heatmap layer."""
        if ais_df.empty: return
        lat_col = "LAT" if "LAT" in ais_df.columns else "latitude"
        lon_col = "LON" if "LON" in ais_df.columns else "longitude"
        
        work = ais_df[[lat_col, lon_col]].dropna()
        if len(work) > max_points:
            work = work.sample(n=max_points, random_state=42)
        self.heatmap_points = work[[lat_col, lon_col]].values.tolist()

    def add_layer(self, scale: str, name: str, data: dict, batch: str = "Default"):
        """Adds a results layer (macro/meso/route) to the dashboard."""
        target = {
            "macro": self.plsn_layers,
            "meso": self.nlsn_layers,
            "route": self.rlsn_layers
        }.get(scale.lower())
        
        if target is not None:
            target[name] = {**data, "batch": batch}

    def generate_dashboard(self, center: tuple = (15.0, 85.0)):
        """Generates the final HTML dashboard."""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Simplified template for modular use
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AIS Modular Analysis Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; background: #020617; color: #f8fafc; font-family: sans-serif; }}
        header {{ background: #0f172a; padding: 10px 20px; border-bottom: 1px solid #1e293b; }}
        #map {{ height: calc(100vh - 60px); width: 100%; }}
    </style>
</head>
<body>
    <header><h1>AIS Modular Analysis - Port Discovery</h1></header>
    <div id="map"></div>
    <script>
        const map = L.map('map').setView([{center[0]}, {center[1]}], 4);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        const plsnSets = {json.dumps(self.plsn_layers)};
        const nlsnSets = {json.dumps(self.nlsn_layers)};
        const rlsnSets = {json.dumps(self.rlsn_layers)};

        const layersControl = L.control.layers(null, {{}}, {{collapsed: false}}).addTo(map);

        function addSet(sets, label, color) {{
            Object.entries(sets).forEach(([name, data]) => {{
                const group = L.layerGroup();
                if (data.nodes) data.nodes.forEach(n => L.circleMarker([n.lat, n.lon], {{radius: 5, color: color}}).addTo(group));
                if (data.edges) data.edges.forEach(e => L.polyline([[e.src_lat, e.src_lon], [e.dst_lat, e.dst_lon]], {{color: color, weight: 1, opacity: 0.5}}).addTo(group));
                if (data.routes) L.geoJSON(data.routes, {{style: {{color: color, weight: 2}}}}).addTo(group);
                if (data.boundaries) L.geoJSON(data.boundaries, {{style: {{color: 'white', weight: 1, dashArray: '5,5', fillOpacity: 0.1}}}}).addTo(group);
                layersControl.addOverlay(group, `[${{data.batch}}] ${{name}}`);
                group.addTo(map);
            }});
        }}

        addSet(plsnSets, "Macro", "#3b82f6");
        addSet(nlsnSets, "Meso", "#10b981");
        addSet(rlsnSets, "Route", "#38bdf8");
    </script>
</body>
</html>
"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info(f"Dashboard generated: {self.output_file}")
