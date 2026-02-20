import pandas as pd
import geopandas as gpd
import logging
import os
from shapely.geometry import shape

class PLSNGenerator:
    """
    Generates the Port-Level Shipping Network (Nodes and Edges).
    Outputs results to files.
    """
    def __init__(self, output_dir):
        """
        Initialize network generator.
        :param output_dir: Directory to save outputs.
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise KeyError(f"Required column not found. Tried: {candidates}")
        return None

    def export_nodes_and_boundaries(self, boundaries_list, clustered_df=None):
        """
        Exports port nodes (centroids) to CSV and boundaries to GeoJSON.
        :param boundaries_list: List of dicts [{'cluster_id': int, 'geometry': Polygon}]
        """
        self.logger.info("Exporting nodes and boundaries...")
        
        if not boundaries_list:
            self.logger.warning("No boundaries to export.")
            return

        # Prepare GeoDataFrame
        gdf = gpd.GeoDataFrame(boundaries_list)
        # Set CRS to WGS84 (EPSG:4326) assuming Lat/Lon data
        gdf.set_crs(epsg=4326, inplace=True)
        
        # Save Boundaries to GeoJSON
        boundary_path = os.path.join(self.output_dir, "boundaries.geojson")
        try:
            gdf.to_file(boundary_path, driver='GeoJSON')
            self.logger.info(f"Boundaries saved to {boundary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save GeoJSON: {e}")
            
        # Calculate Nodes (Centroids)
        # Warning: Centroid in 4326 is geometric, not projected, but sufficient for approx location.
        cluster_point_counts = {}
        if clustered_df is not None and "cluster_id" in clustered_df.columns:
            cluster_point_counts = (
                clustered_df[clustered_df["cluster_id"] != -1]
                .groupby("cluster_id")
                .size()
                .to_dict()
            )

        nodes_data = []
        for _, row in gdf.iterrows():
            centroid = row['geometry'].centroid
            nodes_data.append({
                'port_id': row['cluster_id'],
                'lat': centroid.y,
                'lon': centroid.x,
                'area_deg2': row['geometry'].area,  # Area in square degrees
                'stationary_points': int(cluster_point_counts.get(row['cluster_id'], 0)),
            })
            
        nodes_df = pd.DataFrame(nodes_data)
        nodes_path = os.path.join(self.output_dir, "nodes.csv")
        try:
            nodes_df.to_csv(nodes_path, index=False)
            self.logger.info(f"Nodes saved to {nodes_path}")
        except Exception as e:
            self.logger.error(f"Failed to save Nodes CSV: {e}")

        return nodes_df

    def export_edges(self, clustered_df: pd.DataFrame, valid_port_ids: set[int] | None = None):
        """
        Build directed weighted edges from vessel port-call transitions.
        A transition is counted when cluster_id changes for a vessel in time order.
        """
        self.logger.info("Generating PLSN edges from clustered trajectories...")
        if clustered_df is None or clustered_df.empty:
            self.logger.warning("Clustered dataframe is empty. No edges generated.")
            return pd.DataFrame(columns=["source_port", "target_port", "transition_count", "unique_vessels"])

        mmsi_col = self._pick_column(clustered_df, ["MMSI"])
        time_col = self._pick_column(clustered_df, ["BASEDATETIME", "Timestamp", "DATETIME"], required=False)

        work = clustered_df.copy()
        work = work[work["cluster_id"] != -1].copy()
        if valid_port_ids is not None:
            work = work[work["cluster_id"].isin(valid_port_ids)].copy()

        if work.empty:
            self.logger.warning("No in-port points available for edge generation.")
            return pd.DataFrame(columns=["source_port", "target_port", "transition_count", "unique_vessels"])

        if time_col:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work = work[work[time_col].notna()].copy()
            work.sort_values([mmsi_col, time_col], inplace=True)
        else:
            work.sort_values([mmsi_col], inplace=True)

        # Remove repeated consecutive stays in the same port for each vessel.
        work["prev_cluster"] = work.groupby(mmsi_col)["cluster_id"].shift(1)
        work = work[(work["prev_cluster"].isna()) | (work["prev_cluster"] != work["cluster_id"])].copy()

        work["next_cluster"] = work.groupby(mmsi_col)["cluster_id"].shift(-1)
        transitions = work[
            work["next_cluster"].notna() & (work["cluster_id"] != work["next_cluster"])
        ][[mmsi_col, "cluster_id", "next_cluster"]].copy()

        if transitions.empty:
            self.logger.warning("No inter-port transitions found.")
            edges_df = pd.DataFrame(columns=["source_port", "target_port", "transition_count", "unique_vessels"])
        else:
            transitions["next_cluster"] = transitions["next_cluster"].astype(int)
            edges_df = (
                transitions.groupby(["cluster_id", "next_cluster"])
                .agg(
                    transition_count=(mmsi_col, "size"),
                    unique_vessels=(mmsi_col, "nunique"),
                )
                .reset_index()
                .rename(columns={"cluster_id": "source_port", "next_cluster": "target_port"})
                .sort_values("transition_count", ascending=False)
                .reset_index(drop=True)
            )

        edges_path = os.path.join(self.output_dir, "edges.csv")
        try:
            edges_df.to_csv(edges_path, index=False)
            self.logger.info("Edges saved to %s (%d rows)", edges_path, len(edges_df))
        except Exception as e:
            self.logger.error(f"Failed to save edges CSV: {e}")

        return edges_df
