import os
import json
import logging

class RLSNVisualizer:
    """
    Dedicated visualizer for standalone RLSN maps.
    Generates high-resolution Leaflet maps showing Gaussian routes and boundaries.
    """
    def __init__(self, output_file, center=[15.0, 85.0], zoom=4):
        self.output_file = output_file
        self.center = center
        self.zoom = zoom
        self.logger = logging.getLogger(__name__)

    def generate_map(self, routes_geojson, boundaries_geojson, heatmap_points=None):
        """
        Generates the standalone RLSN map HTML.
        """
        self.logger.info("Generating standalone RLSN map...")
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        heatmap_js = json.dumps(heatmap_points) if heatmap_points else "[]"
        routes_js = json.dumps(routes_geojson)
        bounds_js = json.dumps(boundaries_geojson)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RLSN Standalone Visualizer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #0f172a; height: 100vh; font-family: sans-serif; }}
        #map {{ width: 100%; height: 100%; }}
        .legend {{ background: rgba(30, 41, 59, 0.9); border: 1px solid #334155; padding: 15px; border-radius: 8px; color: #f8fafc; line-height: 1.5; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
        .header {{ position: absolute; top: 20px; left: 50px; z-index: 1000; background: rgba(30, 41, 59, 0.9); padding: 10px 20px; border-radius: 8px; border: 1px solid #334155; color: #38bdf8; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">Route-Level Shipping Network (RLSN) Extraction</div>
    <div id="map"></div>
    <script>
        const map = L.map('map').setView([{self.center[0]}, {self.center[1]}], {self.zoom});
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
        }}).addTo(map);

        const routes = {routes_js};
        const boundaries = {bounds_js};
        const heatPoints = {heatmap_js};

        const overlays = {{}};

        if (heatPoints.length > 0) {{
            overlays["Background Heatmap"] = L.heatLayer(heatPoints, {{radius: 8, blur: 12, minOpacity: 0.2}}).addTo(map);
        }}

        const boundLayer = L.geoJSON(boundaries, {{
            style: function(f) {{
                return {{ color: '#38bdf8', weight: 1, dashArray: '5, 5', fillOpacity: 0.15 }};
            }},
            onEachFeature: function(f, l) {{
                l.bindPopup(`<b>Channel Boundary</b><br>Source: ${{f.properties.source}}<br>Target: ${{f.properties.target}}<br>Sigma: 3-Sigma Rule (99.7%)`);
            }}
        }}).addTo(map);
        overlays["RLSN Channel Boundaries (3&sigma;)"] = boundLayer;

        const routeLayer = L.geoJSON(routes, {{
            style: function(f) {{
                return {{ color: '#0369a1', weight: 3, opacity: 0.8 }};
            }},
            onEachFeature: function(f, l) {{
                l.bindPopup(`<b>Customary Route (Centroid)</b><br>Source: ${{f.properties.source}}<br>Target: ${{f.properties.target}}`);
            }}
        }}).addTo(map);
        overlays["RLSN Customary Routes (&mu;)"] = routeLayer;

        L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

        // Simple Legend
        const legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function() {{
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 5px;">RLSN Scale Legend</div>
                <div style="display:flex; align-items:center; gap:8px;"><span style="width:15px; height:3px; background:#0369a1; display:inline-block;"></span> Customary Route (&mu;)</div>
                <div style="display:flex; align-items:center; gap:8px;"><span style="width:15px; height:6px; background:rgba(56, 189, 248, 0.3); border:1px dashed #38bdf8; display:inline-block;"></span> Safe Channel (&plusmn;3&sigma;)</div>
                <div style="font-size: 0.7rem; color: #64748b; margin-top: 5px;">* Based on Gaussian Traffic Flow Fitting</div>
            `;
            return div;
        }};
        legend.addTo(map);

    </script>
</body>
</html>
"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info("Standalone RLSN map saved to %s", self.output_file)
