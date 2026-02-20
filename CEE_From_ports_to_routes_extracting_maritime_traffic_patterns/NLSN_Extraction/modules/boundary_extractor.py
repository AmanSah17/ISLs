import pandas as pd
import logging
from shapely.geometry import Polygon, MultiPoint
from shapely.errors import GEOSException

try:
    import alphashape
except ImportError:  # pragma: no cover
    alphashape = None

class BoundaryExtractor:
    """
    Extracts the boundary of a cluster using the Alpha-shape algorithm.
    """
    def __init__(self, alpha=0.01):
        """
        Initialize Boundary Extractor.
        :param alpha: Alpha value for the shape. 
                      Higher alpha -> tighter fit (more concave details).
                      Lower alpha -> closer to Convex Hull.
        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
        self._skipped_degenerate = 0
        self._failed_clusters = 0
        if alphashape is None:
            self.logger.warning(
                "alphashape package not available. Falling back to convex hull boundaries."
            )

    def extract_boundaries(self, df):
        """
        Extracts boundaries for each unique cluster in the dataframe.
        :param df: Input DataFrame with 'LAT', 'LON', 'cluster_id'.
        :return: List of dictionaries [{'cluster_id': int, 'geometry': Polygon}]
        """
        self.logger.info(f"Extracting boundaries with alpha={self.alpha}...")
        boundaries = []
        
        # Get unique cluster IDs (exclude noise -1)
        cluster_ids = df['cluster_id'].unique()
        cluster_ids = [cid for cid in cluster_ids if cid != -1]
        
        grouped = (
            df[df["cluster_id"] != -1]
            .groupby("cluster_id", sort=False)[["LON", "LAT"]]
        )
        for cid, grp in grouped:
            cluster_points = grp.to_numpy()
            
            # Need at least 3 points for a polygon, but alpha shape might need more for stability
            if len(cluster_points) < 4:
                self.logger.debug(f"Skipping cluster {cid}: too few points ({len(cluster_points)}).")
                continue

            # Skip degenerate clusters that are effectively on a point/line; this avoids noisy qhull failures.
            unique_xy = pd.DataFrame(cluster_points, columns=["x", "y"]).drop_duplicates()
            if len(unique_xy) < 4:
                self._skipped_degenerate += 1
                continue
            if unique_xy["x"].nunique() < 2 or unique_xy["y"].nunique() < 2:
                self._skipped_degenerate += 1
                continue
                
            try:
                if alphashape is not None:
                    # Note: alphashape takes (x, y) points -> (LON, LAT)
                    hull = alphashape.alphashape(cluster_points, self.alpha)
                else:
                    hull = MultiPoint(cluster_points).convex_hull
                
                if hull.is_empty:
                    self.logger.warning(f"Cluster {cid} resulted in empty hull.")
                    continue
                if not hull.is_valid:
                    hull = hull.buffer(0)
                    if hull.is_empty:
                        continue
                    
                if hull.geom_type == 'Polygon':
                    boundaries.append({'cluster_id': int(cid), 'geometry': hull})
                elif hull.geom_type == 'MultiPolygon':
                    # Simplify MultiPolygon to largest component or keep as is
                    # Here we keep the largest polygon for simplicity
                    largest_poly = max(hull.geoms, key=lambda a: a.area)
                    if not largest_poly.is_valid:
                        largest_poly = largest_poly.buffer(0)
                    boundaries.append({'cluster_id': int(cid), 'geometry': largest_poly})
                else:
                    self.logger.warning(f"Cluster {cid} resulted in unexpected geometry: {hull.geom_type}")
                    
            except GEOSException as e:
                self._failed_clusters += 1
                self.logger.debug("GEOS error on cluster %s: %s", str(cid), str(e))
            except Exception as e:
                self._failed_clusters += 1
                self.logger.debug("Boundary extraction error on cluster %s: %s", str(cid), str(e))
        
        if self._skipped_degenerate > 0:
            self.logger.info("Skipped %d degenerate clusters before alpha-shape.", self._skipped_degenerate)
        if self._failed_clusters > 0:
            self.logger.info("Boundary extraction failed for %d clusters (suppressed detailed stack traces).", self._failed_clusters)
        self.logger.info(f"Extracted {len(boundaries)} valid boundaries.")
        return boundaries
