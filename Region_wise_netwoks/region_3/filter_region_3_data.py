import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import os
import json

# --- Configuration ---
REGION_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(REGION_ROOT, "region_3.geojson")
GLOBAL_AIS_PATH = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"
OUTPUT_PARQUET = os.path.join(REGION_ROOT, "region_3_ais.parquet")

def filter_data():
    print(f"Loading Region 3 polygon from {GEOJSON_PATH}...")
    with open(GEOJSON_PATH, 'r') as f:
        geojson_data = json.load(f)
    geom = shape(geojson_data['features'][0]['geometry'])
    
    print(f"Loading global AIS data...")
    df = pd.read_parquet(GLOBAL_AIS_PATH)
    
    minx, miny, maxx, maxy = geom.bounds
    mask = (df['LON'] >= minx) & (df['LON'] <= maxx) & (df['LAT'] >= miny) & (df['LAT'] <= maxy)
    df_filtered = df[mask].copy()
    
    gdf = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered.LON, df_filtered.LAT), crs="EPSG:4326")
    df_final = gdf[gdf.geometry.within(geom)].drop(columns='geometry')
    
    print(f"Final rows in Region 3: {len(df_final)}")
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print("Done.")

if __name__ == "__main__":
    filter_data()
