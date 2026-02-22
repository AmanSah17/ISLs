import duckdb

PARQUET_PATH = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"

print("Querying Parquet file bounds and count...")
try:
    con = duckdb.connect(':memory:')
    q = f"""
        SELECT 
            MIN(LON) as min_lon, MAX(LON) as max_lon, 
            MIN(LAT) as min_lat, MAX(LAT) as max_lat,
            COUNT(*) as total_rows
        FROM '{PARQUET_PATH}'
    """
    res = con.execute(q).fetchone()
    print(f"Total Rows: {res[4]}")
    print(f"Bounds: LON [{res[0]}, {res[1]}], LAT [{res[2]}, {res[3]}]")
    
except Exception as e:
    print(f"Error: {e}")
