import logging
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

class RLSNGenerator:
    """
    Modular RLSN Generator for standalone extraction.
    Performs slice-based Gaussian traffic flow fitting between waypoints.
    """

    def __init__(self, output_dir, num_slices=10, search_radius=0.1, sigma_mult=3):
        self.output_dir = output_dir
        self.num_slices = num_slices
        self.search_radius = search_radius
        self.sigma_mult = sigma_mult
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_rlsn(self, ais_df, nodes_df, edges_df):
        """
        Extracts customary routes and boundaries.
        :param ais_df: DataFrame of AIS points (LAT, LON).
        :param nodes_df: DataFrame of NLSN nodes (port_id, lat, lon).
        :param edges_df: DataFrame of NLSN edges (source_port, target_port).
        """
        self.logger.info("Starting standalone RLSN extraction for %d edges...", len(edges_df))
        
        nodes_dict = nodes_df.set_index("port_id")[["lat", "lon"]].to_dict("index")
        rlsn_routes = []
        rlsn_boundaries = []

        for idx, edge in edges_df.iterrows():
            src_id, tgt_id = int(edge["source_port"]), int(edge["target_port"])
            if src_id not in nodes_dict or tgt_id not in nodes_dict:
                continue

            src, tgt = nodes_dict[src_id], nodes_dict[tgt_id]
            A, B = np.array([src["lon"], src["lat"]]), np.array([tgt["lon"], tgt["lat"]])
            V = B - A
            length = np.linalg.norm(V)
            if length < 1e-7: continue
            
            V_norm = V / length
            N = np.array([-V_norm[1], V_norm[0]])  # Perpendicular normal
            edge_slices = []
            
            for i in range(1, self.num_slices + 1):
                fraction = i / (self.num_slices + 1)
                C_i = A + fraction * V
                
                # Spatial filtering
                mask = (
                    (ais_df["LON"] >= C_i[0] - self.search_radius) &
                    (ais_df["LON"] <= C_i[0] + self.search_radius) &
                    (ais_df["LAT"] >= C_i[1] - self.search_radius) &
                    (ais_df["LAT"] <= C_i[1] + self.search_radius)
                )
                nearby_points = ais_df[mask][["LON", "LAT"]].to_numpy()
                if len(nearby_points) < 10: continue

                # Projection onto Normal segment
                P_minus_C = nearby_points - C_i
                dist_along = P_minus_C @ V_norm
                strip_width = length / (self.num_slices * 1.5) # Overlap slightly for smoothness
                strip_mask = np.abs(dist_along) < strip_width
                valid_offsets = (P_minus_C[strip_mask] @ N)
                
                if len(valid_offsets) < 5: continue

                # Gaussian Fitting
                mu, sigma = norm.fit(valid_offsets)
                edge_slices.append({
                    "center": C_i + mu * N,
                    "sigma": sigma,
                    "normal": N
                })

            if not edge_slices: continue

            # Geometry Packaging
            route_coords = [s["center"] for s in edge_slices]
            left_boundary = [s["center"] + self.sigma_mult * s["sigma"] * s["normal"] for s in edge_slices]
            right_boundary = [s["center"] - self.sigma_mult * s["sigma"] * s["normal"] for s in edge_slices]
            
            rlsn_routes.append({
                "source": src_id, "target": tgt_id,
                "geometry": {"type": "LineString", "coordinates": [c.tolist() for c in route_coords]}
            })

            poly_coords = left_boundary + right_boundary[::-1]
            poly_coords.append(left_boundary[0])
            rlsn_boundaries.append({
                "source": src_id, "target": tgt_id,
                "geometry": {"type": "Polygon", "coordinates": [[c.tolist() for c in poly_coords]]}
            })

        # Save to GeoJSON
        self._save_geojson("rlsn_routes.geojson", rlsn_routes)
        self._save_geojson("rlsn_boundaries.geojson", rlsn_boundaries)
        return rlsn_routes, rlsn_boundaries

    def _save_geojson(self, filename, features):
        path = os.path.join(self.output_dir, filename)
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"source": f["source"], "target": f["target"]},
                "geometry": f["geometry"]
            } for f in features]
        }
        with open(path, "w") as f:
            json.dump(geojson, f)
        self.logger.info("Saved RLSN data to %s", path)
