import json
import logging

import pandas as pd


class PLSNVisualizer:
    """
    Generates a standalone HTML dashboard using Leaflet (no folium dependency).
    """

    def __init__(self, output_file, sample_size=200000):
        self.output_file = output_file
        self.sample_size = sample_size
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _sample_points(df: pd.DataFrame, lat_col: str, lon_col: str, sample_size: int) -> list[list[float]]:
        if df.empty:
            return []
        work = df[[lat_col, lon_col]].dropna()
        if len(work) > sample_size:
            work = work.sample(n=sample_size, random_state=42)
        # Round + float32 to reduce HTML payload and Python memory during JSON serialization.
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce").astype("float32").round(5)
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce").astype("float32").round(5)
        return work[[lat_col, lon_col]].dropna().values.tolist()

    def generate_plsn_dashboard(
        self,
        full_df: pd.DataFrame,
        clustered_df: pd.DataFrame,
        boundaries_list: list[dict],
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ):
        self.logger.info("Generating standalone PLSN dashboard HTML...")

        lat_col = "LAT" if "LAT" in full_df.columns else "Latitude"
        lon_col = "LON" if "LON" in full_df.columns else "Longitude"

        all_points = self._sample_points(full_df, lat_col, lon_col, self.sample_size)
        in_port_df = clustered_df[clustered_df["cluster_id"] != -1] if "cluster_id" in clustered_df.columns else clustered_df
        in_points = self._sample_points(in_port_df, lat_col, lon_col, self.sample_size)

        geojson_features = []
        for item in boundaries_list:
            geojson_features.append(
                {
                    "type": "Feature",
                    "properties": {"cluster_id": int(item["cluster_id"])},
                    "geometry": item["geometry"].__geo_interface__,
                }
            )
        boundaries_geojson = {"type": "FeatureCollection", "features": geojson_features}

        nodes = []
        node_lookup = {}
        for row in nodes_df.itertuples(index=False):
            node = {
                "port_id": int(row.port_id),
                "lat": float(row.lat),
                "lon": float(row.lon),
                "stationary_points": int(getattr(row, "stationary_points", 0)),
                "area_deg2": float(getattr(row, "area_deg2", 0.0)),
            }
            nodes.append(node)
            node_lookup[node["port_id"]] = (node["lat"], node["lon"])

        edges = []
        for row in edges_df.itertuples(index=False):
            src = int(row.source_port)
            dst = int(row.target_port)
            if src not in node_lookup or dst not in node_lookup:
                continue
            edges.append(
                {
                    "source_port": src,
                    "target_port": dst,
                    "transition_count": int(row.transition_count),
                    "unique_vessels": int(row.unique_vessels),
                    "source_lat": node_lookup[src][0],
                    "source_lon": node_lookup[src][1],
                    "target_lat": node_lookup[dst][0],
                    "target_lon": node_lookup[dst][1],
                }
            )

        top_ports = sorted(nodes, key=lambda x: x["stationary_points"], reverse=True)[:10]
        top_edges = sorted(edges, key=lambda x: x["transition_count"], reverse=True)[:10]

        center_lat = float(nodes_df["lat"].mean()) if not nodes_df.empty else 0.0
        center_lon = float(nodes_df["lon"].mean()) if not nodes_df.empty else 0.0
        total_transitions = int(edges_df["transition_count"].sum()) if not edges_df.empty else 0

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PLSN Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    body {{ margin: 0; font-family: 'Segoe UI', sans-serif; }}
    #map {{ height: 100vh; width: 100vw; }}
    #panel {{
      position: fixed; top: 16px; right: 16px; width: 360px; max-height: 88vh; overflow: auto;
      z-index: 9999; background: rgba(255,255,255,0.95); border: 1px solid #d1d5db;
      border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.2); padding: 14px; font-size: 13px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 4px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
    h3, h4 {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="panel">
    <h3>PLSN Dashboard</h3>
    <div><b>Ports:</b> {len(nodes)}</div>
    <div><b>Directed Edges:</b> {len(edges)}</div>
    <div><b>Total Port Transitions:</b> {total_transitions}</div>
    <div><b>Stationary Points Used:</b> {sum([x["stationary_points"] for x in nodes])}</div>
    <h4>Top Ports</h4>
    <table><thead><tr><th>Port</th><th>Points</th></tr></thead><tbody>
      {"".join([f"<tr><td>{x['port_id']}</td><td>{x['stationary_points']}</td></tr>" for x in top_ports]) or "<tr><td colspan='2'>No data</td></tr>"}
    </tbody></table>
    <h4>Top Routes</h4>
    <table><thead><tr><th>Edge</th><th>Transitions</th></tr></thead><tbody>
      {"".join([f"<tr><td>{x['source_port']}→{x['target_port']}</td><td>{x['transition_count']}</td></tr>" for x in top_edges]) or "<tr><td colspan='2'>No data</td></tr>"}
    </tbody></table>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
  <script>
    const map = L.map('map').setView([{center_lat}, {center_lon}], {5 if nodes else 2});
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const allPoints = {json.dumps(all_points)};
    const inPoints = {json.dumps(in_points)};
    const boundaries = {json.dumps(boundaries_geojson)};
    const nodes = {json.dumps(nodes)};
    const edges = {json.dumps(edges)};

    if (allPoints.length) {{
      L.heatLayer(allPoints, {{radius: 8, blur: 12, minOpacity: 0.2}}).addTo(map);
    }}
    if (inPoints.length) {{
      L.heatLayer(inPoints, {{radius: 7, blur: 10, minOpacity: 0.35, gradient: {{0.4: '#f59e0b', 0.7: '#ef4444', 1.0: '#7f1d1d'}}}}).addTo(map);
    }}

    L.geoJSON(boundaries, {{
      style: function() {{ return {{ color: '#b91c1c', weight: 2, fillColor: '#dc2626', fillOpacity: 0.12 }}; }},
      onEachFeature: function(feature, layer) {{
        layer.bindTooltip('Port Cluster ' + feature.properties.cluster_id);
      }}
    }}).addTo(map);

    nodes.forEach(function(n) {{
      const radius = Math.max(3, Math.min(12, Math.sqrt(Math.max(n.stationary_points, 1)) / 2));
      L.circleMarker([n.lat, n.lon], {{
        radius: radius, color: '#0f172a', fillColor: '#22c55e', fillOpacity: 0.9
      }}).addTo(map).bindPopup(
        `Port ${{n.port_id}}<br>Stationary points: ${{n.stationary_points}}<br>Area (deg²): ${{n.area_deg2.toFixed(6)}}`
      );
    }});

    const maxTransitions = edges.length ? Math.max(...edges.map(e => e.transition_count)) : 1;
    edges.forEach(function(e) {{
      const w = Math.max(1, 6 * (e.transition_count / maxTransitions));
      L.polyline([[e.source_lat, e.source_lon], [e.target_lat, e.target_lon]], {{
        color: '#1d4ed8', weight: w, opacity: 0.65
      }}).addTo(map).bindTooltip(
        `${{e.source_port}} → ${{e.target_port}} | transitions=${{e.transition_count}}, vessels=${{e.unique_vessels}}`
      );
    }});
  </script>
</body>
</html>
"""

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info("Saved PLSN dashboard HTML to %s", self.output_file)
