"""
convert_data.py — Convert parquet AIS data to GLASSEAS CSV format.

Reads the Gulf of Mexico parquet file, filters to original (non-interpolated)
positions, maps columns to GLASSEAS expected format, and saves as CSV.
"""

import pandas as pd
import sys
import os

PARQUET_PATH = r"F:\PyTorch_GPU\AIS_trajectory_forecasting\Data\region_1_q1_merged_renamed.parquet"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dataset.csv")

# AIS numeric vessel type codes → readable names
VESSEL_TYPE_MAP = {
    (20, 29): "WIG",
    (30, 39): "Fishing",
    (40, 49): "HSC",
    (50, 59): "Tug/Special",
    (60, 69): "Passenger",
    (70, 79): "Cargo",
    (80, 89): "Tanker",
    (90, 99): "Other",
}


def map_vessel_type(code):
    try:
        c = int(code)
    except (ValueError, TypeError):
        return "Unknown"
    for (lo, hi), name in VESSEL_TYPE_MAP.items():
        if lo <= c <= hi:
            return name
    return "Unknown"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading parquet file...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Total rows: {len(df):,}")
    
    # Filter to original (non-interpolated) positions only
    df_orig = df[df['is_interpolated'] == False].copy()
    print(f"  Original (non-interpolated) rows: {len(df_orig):,}")
    
    # Map vessel type codes to readable names
    print("Mapping vessel types...")
    df_orig['ShipType'] = df_orig['VESSEL_TYPE'].apply(map_vessel_type)
    
    # Format timestamp
    df_orig['Timestamp'] = pd.to_datetime(df_orig['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Fill missing COG with 0
    df_orig['COG'] = df_orig['COG'].fillna(0.0)
    
    # Build GLASSEAS-format DataFrame
    # Columns: MMSI,IMO,LATITUDE,LONGITUDE,COG,HEADING,SOG,TIMESTAMP,NAME,SHIP_TYPE,DESTINATION,ANNOTATION
    glasseas_df = pd.DataFrame({
        'MMSI': df_orig['MMSI'].astype(int),
        'IMO': -1,
        'LATITUDE': df_orig['Latitude'],
        'LONGITUDE': df_orig['Longitude'],
        'COG': df_orig['COG'].round(6),
        'HEADING': df_orig['COG'].round(6),  # Use COG as heading (no separate heading field)
        'SOG': (df_orig['SOG'] * 10).round(0).astype(int),  # GLASSEAS stores SOG * 10
        'TIMESTAMP': df_orig['Timestamp'],
        'NAME': df_orig['MMSI'].astype(str),
        'SHIP_TYPE': df_orig['ShipType'],
        'DESTINATION': 'NA',
        'ANNOTATION': 'NULL',
    })
    
    print(f"Saving to {OUTPUT_CSV}...")
    glasseas_df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("\n=== Data Summary ===")
    print(f"Total positions: {len(glasseas_df):,}")
    print(f"Unique vessels: {glasseas_df['MMSI'].nunique():,}")
    print(f"\nVessel type distribution:")
    print(glasseas_df['SHIP_TYPE'].value_counts().to_string())
    print(f"\nLon range: [{glasseas_df['LONGITUDE'].min():.4f}, {glasseas_df['LONGITUDE'].max():.4f}]")
    print(f"Lat range: [{glasseas_df['LATITUDE'].min():.4f}, {glasseas_df['LATITUDE'].max():.4f}]")
    print(f"\nTime range: {glasseas_df['TIMESTAMP'].min()} to {glasseas_df['TIMESTAMP'].max()}")
    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"File size: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
