import duckdb
import geopandas as gpd
import json
import pandas as pd
import os

DB_PATH = "f:/PyTorch_GPU/ISLs/AIS_Modular_Processor/data/ais_platform.duckdb"
PARQUET_PATH = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"

out_f = open("debug_results_2.txt", "w")

def log(msg):
    print(msg)
    out_f.write(str(msg) + "\n")

con = duckdb.connect(DB_PATH, read_only=True)
res = con.execute("SELECT id, name, geojson FROM spatial_regions WHERE id = 9").fetchone()

gj = json.loads(res[2])
temp_gj = "temp_region_9.geojson"
with open(temp_gj, "w") as f:
    json.dump(gj, f)

region_gdf = gpd.read_file(temp_gj)
bounds = region_gdf.total_bounds

q = f"""
    SELECT LAT as lat, LON as lon, MMSI as mmsi FROM '{PARQUET_PATH}'
    WHERE LON >= {bounds[0]} AND LON <= {bounds[2]}
      AND LAT >= {bounds[1]} AND LAT <= {bounds[3]}
"""
ais_df = duckdb.query(q).df()

if ais_df.empty:
    log("duckdb dataframe is empty")
else:
    log(f"Fetched {len(ais_df)} points from DuckDB.")
    ais_gdf = gpd.GeoDataFrame(
        ais_df, 
        geometry=gpd.points_from_xy(ais_df['lon'], ais_df['lat']),
        crs="EPSG:4326"
    )
    log(f"ais_gdf bounds: {ais_gdf.total_bounds}")
    log(f"region_gdf bounds: {region_gdf.total_bounds}")
    log(f"region_gdf CRS: {region_gdf.crs}")

    clipped_gdf = gpd.sjoin(ais_gdf, region_gdf, how="inner", predicate="within")
    log(f"Points after exact polygon clip (within): {len(clipped_gdf)}")
    
    clipped_gdf_intersect = gpd.sjoin(ais_gdf, region_gdf, how="inner", predicate="intersects")
    log(f"Points after exact polygon clip (intersects): {len(clipped_gdf_intersect)}")

out_f.close()
