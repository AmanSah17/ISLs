import duckdb
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DUCKDB_PATH

def setup_duckdb():
    print(f"--- Setting up DuckDB schema in {DUCKDB_PATH} ---")
    
    # Connect to DuckDB
    con = duckdb.connect(DUCKDB_PATH)
    
    # Load and install spatial extension
    print("Installing spatial extension...")
    con.execute("INSTALL spatial;")
    print("Loading spatial extension...")
    con.execute("LOAD spatial;")
    
    print("Creating tables...")
    # 1. Users Table
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_users_id;
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER DEFAULT nextval('seq_users_id') PRIMARY KEY,
        username VARCHAR NOT NULL UNIQUE,
        email VARCHAR NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT true
    );
    """)

    # 2. Analysis Runs Table
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_analysis_runs_id;
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id INTEGER DEFAULT nextval('seq_analysis_runs_id') PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        run_name VARCHAR,
        status VARCHAR DEFAULT 'pending',
        config JSON,
        metadata_info JSON,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    """)

    # 3. Spatial Regions Table
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_spatial_regions_id;
    CREATE TABLE IF NOT EXISTS spatial_regions (
        id INTEGER DEFAULT nextval('seq_spatial_regions_id') PRIMARY KEY,
        run_id INTEGER REFERENCES analysis_runs(id),
        user_id INTEGER REFERENCES users(id),
        name VARCHAR,
        geojson JSON,
        geom GEOMETRY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 4. Port Nodes (PLSN)
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_port_nodes_id;
    CREATE TABLE IF NOT EXISTS port_nodes (
        id INTEGER DEFAULT nextval('seq_port_nodes_id') PRIMARY KEY,
        run_id INTEGER REFERENCES analysis_runs(id),
        cluster_id INTEGER,
        weight INTEGER,
        geom GEOMETRY
    );
    """)

    # 5. Shipping Edges (NLSN)
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_shipping_edges_id;
    CREATE TABLE IF NOT EXISTS shipping_edges (
        id INTEGER DEFAULT nextval('seq_shipping_edges_id') PRIMARY KEY,
        run_id INTEGER REFERENCES analysis_runs(id),
        source_id INTEGER,
        target_id INTEGER,
        weight INTEGER,
        geom GEOMETRY
    );
    """)

    # 6. Vessel Trajectories (RLSN)
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_vessel_trajectories_id;
    CREATE TABLE IF NOT EXISTS vessel_trajectories (
        id INTEGER DEFAULT nextval('seq_vessel_trajectories_id') PRIMARY KEY,
        run_id INTEGER REFERENCES analysis_runs(id),
        mmsi VARCHAR,
        vessel_type VARCHAR,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        point_count INTEGER,
        geom GEOMETRY
    );
    """)

    # 7. Tuning Results
    con.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_tuning_results_id;
    CREATE TABLE IF NOT EXISTS tuning_results (
        id INTEGER DEFAULT nextval('seq_tuning_results_id') PRIMARY KEY,
        run_id INTEGER REFERENCES analysis_runs(id),
        k_value INTEGER,
        r_value DOUBLE,
        num_ports INTEGER,
        silhouette_score DOUBLE,
        processing_time DOUBLE
    );
    """)

    print("All DuckDB tables created successfully.")
    con.close()

if __name__ == "__main__":
    setup_duckdb()
