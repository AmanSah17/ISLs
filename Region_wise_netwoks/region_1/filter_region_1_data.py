import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import json
import os
import sys

def filter_data():
    source_parquet = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"
    geojson_path = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\Region_wise_netwoks\region_1\region_1.geojson"
    output_parquet = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\Region_wise_netwoks\region_1\region_1_ais.parquet"

    print(f"Loading GeoJSON from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    # The GeoJSON might be nested, let's navigate to the feature
    if "features" in geojson_data:
        geom = shape(geojson_data["features"][0]["geometry"])
    else:
        geom = shape(geojson_data["geometry"])

    print(f"Loading source parquet (this might take a while)...")
    df = pd.read_parquet(source_parquet)
    print(f"Total rows in source: {len(df)}")

    print(f"Converting to GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.LON, df.LAT),
        crs="EPSG:4326"
    )

    print(f"Filtering data by region boundary...")
    # Use spatial join or within check
    # For large datasets, within is faster if we use spatial index
    region_gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs="EPSG:4326")
    filtered_gdf = gpd.sjoin(gdf, region_gdf, how="inner", predicate='within')

    print(f"Filtered rows: {len(filtered_gdf)}")

    # Drop geometry column to save as parquet (preserving original format)
    # and drop index_right from sjoin
    filtered_df = filtered_gdf.drop(columns=['geometry', 'index_right'])

    print(f"Saving filtered data to {output_parquet}...")
    filtered_df.to_parquet(output_parquet)
    print("Done!")

if __name__ == "__main__":
    filter_data()
