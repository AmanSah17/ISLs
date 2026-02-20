from __future__ import annotations

import logging
import os

import geopandas as gpd
import pandas as pd


class NLSNGenerator:
    """
    Build and export Node-Level Shipping Network (NLSN) artifacts:
    - feature points
    - node centroids + optional boundaries
    - directed transition edges
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise KeyError(f"Required column not found. Tried: {candidates}")
        return None

    def export_feature_points(self, feature_df: pd.DataFrame, filename: str = "feature_points.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        feature_df.to_csv(path, index=False)
        self.logger.info("Feature points saved to %s (%d rows)", path, len(feature_df))
        return path

    def export_nodes_and_boundaries(
        self,
        boundaries_list: list[dict] | None,
        clustered_feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.logger.info("Exporting NLSN nodes and boundaries...")
        nodes_path = os.path.join(self.output_dir, "nodes.csv")
        boundaries_path = os.path.join(self.output_dir, "boundaries.geojson")

        non_noise = clustered_feature_df[clustered_feature_df["cluster_id"] != -1].copy()
        if non_noise.empty:
            nodes_df = pd.DataFrame(columns=["port_id", "lat", "lon", "feature_points", "area_deg2"])
            nodes_df.to_csv(nodes_path, index=False)
            self.logger.warning("No non-noise clusters. Empty nodes written to %s", nodes_path)
            return nodes_df

        point_counts = non_noise.groupby("cluster_id").size().to_dict()

        if boundaries_list:
            gdf = gpd.GeoDataFrame(boundaries_list)
            gdf.set_crs(epsg=4326, inplace=True)
            try:
                gdf.to_file(boundaries_path, driver="GeoJSON")
                self.logger.info("Boundaries saved to %s", boundaries_path)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Failed writing boundaries GeoJSON: %s", exc)

            rows = []
            for _, row in gdf.iterrows():
                centroid = row["geometry"].centroid
                rows.append(
                    {
                        "port_id": int(row["cluster_id"]),
                        "lat": float(centroid.y),
                        "lon": float(centroid.x),
                        "feature_points": int(point_counts.get(int(row["cluster_id"]), 0)),
                        "area_deg2": float(row["geometry"].area),
                    }
                )
            nodes_df = pd.DataFrame(rows)
        else:
            # Fallback: direct cluster centroids when no polygon boundaries were extracted.
            nodes_df = (
                non_noise.groupby("cluster_id")[["LAT", "LON"]]
                .mean()
                .reset_index()
                .rename(columns={"cluster_id": "port_id", "LAT": "lat", "LON": "lon"})
            )
            nodes_df["feature_points"] = nodes_df["port_id"].map(point_counts).fillna(0).astype(int)
            nodes_df["area_deg2"] = 0.0

        nodes_df = nodes_df.sort_values("feature_points", ascending=False).reset_index(drop=True)
        nodes_df.to_csv(nodes_path, index=False)
        self.logger.info("Nodes saved to %s (%d rows)", nodes_path, len(nodes_df))
        return nodes_df

    def export_edges(
        self,
        clustered_feature_df: pd.DataFrame,
        valid_port_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        self.logger.info("Generating NLSN directed edges from clustered feature sequences...")
        edges_path = os.path.join(self.output_dir, "edges.csv")

        if clustered_feature_df is None or clustered_feature_df.empty:
            edges_df = pd.DataFrame(columns=["source_port", "target_port", "transition_count", "unique_vessels"])
            edges_df.to_csv(edges_path, index=False)
            return edges_df

        mmsi_col = self._pick_column(clustered_feature_df, ["MMSI"])
        track_col = self._pick_column(clustered_feature_df, ["trajectory_id", "track_id"], required=False)
        time_col = self._pick_column(clustered_feature_df, ["BASEDATETIME", "Timestamp", "DATETIME"], required=False)
        order_col = self._pick_column(clustered_feature_df, ["feature_order"], required=False)

        work = clustered_feature_df.copy()
        work = work[work["cluster_id"] != -1].copy()
        if valid_port_ids is not None:
            work = work[work["cluster_id"].isin(valid_port_ids)].copy()

        if work.empty:
            edges_df = pd.DataFrame(columns=["source_port", "target_port", "transition_count", "unique_vessels"])
            edges_df.to_csv(edges_path, index=False)
            self.logger.warning("No non-noise points left for edge generation.")
            return edges_df

        if track_col is None:
            work["trajectory_id"] = work[mmsi_col].astype(str) + "_0"
            track_col = "trajectory_id"

        if time_col:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        sort_cols = [mmsi_col, track_col]
        if order_col:
            sort_cols.append(order_col)
        if time_col:
            sort_cols.append(time_col)
        work.sort_values(sort_cols, inplace=True)

        # Remove repeated consecutive occurrences of the same node.
        work["prev_cluster"] = work.groupby([mmsi_col, track_col])["cluster_id"].shift(1)
        dedup = work[(work["prev_cluster"].isna()) | (work["prev_cluster"] != work["cluster_id"])].copy()
        dedup["next_cluster"] = dedup.groupby([mmsi_col, track_col])["cluster_id"].shift(-1)
        transitions = dedup[
            dedup["next_cluster"].notna() & (dedup["cluster_id"] != dedup["next_cluster"])
        ][[mmsi_col, "cluster_id", "next_cluster"]].copy()

        if transitions.empty:
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

        edges_df.to_csv(edges_path, index=False)
        self.logger.info("Edges saved to %s (%d rows)", edges_path, len(edges_df))
        return edges_df
