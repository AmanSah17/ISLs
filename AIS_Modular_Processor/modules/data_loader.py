import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point
import logging
import os
import json

class AISDataLoader:
    """
    Modular loader for AIS data (Parquet/CSV) and GeoJSON regions.
    """
    def __init__(self):
        self.logger = logging.getLogger("AIS_DataLoader")
        # Standard column mapping
        self.col_map = {
            "mmsi": ["mmsi", "MMSI", "vessel_id", "vessel_name"],
            "lat": ["latitude", "LAT", "lat", "latitude_approximate"],
            "lon": ["longitude", "LON", "lon", "longitude_approximate"],
            "sog": ["sog", "SOG", "speed", "sog_approximate"],
            "navstatus": ["navigational_status", "NAVSTATUS", "status"],
            "basedatetime": ["timestamp_updated", "BASEDATETIME", "datetime", "timestamp_created", "TIMESTAMP", "BaseDateTime"]
        }

    def _normalize_columns(self, columns):
        """Standardizes column names based on map for DuckDB query generation."""
        rename_map = {}
        for target, candidates in self.col_map.items():
            for cand in candidates:
                if cand in columns.tolist():
                    rename_map[cand] = target.lower()
                    break
        return rename_map

    def clip_by_region(self, file_path, geojson_path):
        """Uses DuckDB to query massive Parquet/CSV files using spatial bounds instantly."""
        if not os.path.exists(geojson_path) or not os.path.exists(file_path):
            raise FileNotFoundError("AIS or GeoJSON file not found.")

        # 1. Get Bounds from GeoJSON
        region_gdf = gpd.read_file(geojson_path)
        bounds = region_gdf.total_bounds  # [minx, miny, maxx, maxy]
        self.logger.info(f"Using DuckDB bbox pre-filter: {bounds}")
        
        # 2. Setup DuckDB Query
        import duckdb
        con = duckdb.connect(database=':memory:')
        
        # Find raw column names first so we can map them
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".parquet":
            cols = con.execute(f"SELECT * FROM parquet_schema('{file_path}')").df()['name']
        else:
            cols = pd.read_csv(file_path, nrows=0).columns

        rename_map = self._normalize_columns(cols)
        
        # We need the original column names for the SQL query
        raw_lat = next((k for k, v in rename_map.items() if v == 'lat'), None)
        raw_lon = next((k for k, v in rename_map.items() if v == 'lon'), None)
        
        if not raw_lat or not raw_lon:
            raise KeyError("Could not find latitude or longitude columns in file schema.")

        # Build projected SELECT fields with standardized aliases
        select_fields = ", ".join([f'"{raw}" AS {std}' for raw, std in rename_map.items()])

        # Query using bounds directly against the file! This is extremely memory efficient
        query = f"""
            SELECT {select_fields}
            FROM '{file_path}'
            WHERE "{raw_lon}" >= {bounds[0]} AND "{raw_lon}" <= {bounds[2]}
              AND "{raw_lat}" >= {bounds[1]} AND "{raw_lat}" <= {bounds[3]}
        """
        
        self.logger.info("Executing optimized DuckDB query...")
        ais_df = con.execute(query).df()
        con.close()
        
        self.logger.info(f"DuckDB returned {len(ais_df)} Points. Running strict polygon clip...")
        if ais_df.empty: return ais_df

        # 3. Strict polygon clip using GeoPandas
        ais_gdf = gpd.GeoDataFrame(
            ais_df, 
            geometry=gpd.points_from_xy(ais_df['lon'], ais_df['lat']),
            crs="EPSG:4326"
        )
        clipped_gdf = gpd.sjoin(ais_gdf, region_gdf, how="inner", predicate="within")
        if 'index_right' in clipped_gdf.columns:
            clipped_gdf.drop(columns=['index_right'], inplace=True)
            
        clipped_df = pd.DataFrame(clipped_gdf.drop(columns='geometry'))
        self.logger.info(f"Final clipped dataset: {len(clipped_df)} rows.")
        return clipped_df
