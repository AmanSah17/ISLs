import sys
import os

# Ensure project root is on path so 'config' and 'modules' are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from sqlalchemy import text
from modules.db_session import engine
from modules.models import Base
from config import DB_NAME

def setup_database():
    """Enables PostGIS extension and creates all tables in port_geodata."""
    print(f"--- Setting up schema in: {DB_NAME} ---")
    
    # Enable PostGIS and create tables in the existing database
    with engine.connect() as conn:
        print("Enabling PostGIS extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.commit()
        print("PostGIS enabled.")
    
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully.")
    print(f"Tables: users, analysis_runs, spatial_regions, tuning_results, port_nodes, shipping_edges, vessel_trajectories")

if __name__ == "__main__":
    setup_database()
