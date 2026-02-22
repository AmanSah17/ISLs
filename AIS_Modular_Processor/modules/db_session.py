import duckdb
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DUCKDB_PATH

# Global single connection for FastAPI concurrency
_global_con = None

def _get_global_connection():
    global _global_con
    if _global_con is None:
        _global_con = duckdb.connect(DUCKDB_PATH, read_only=False)
        _global_con.execute("INSTALL spatial;")
        _global_con.execute("LOAD spatial;")
    return _global_con

def get_db():
    """Yields a cursor from the global DuckDB connection for FastAPI Depends."""
    con = _get_global_connection()
    cursor = con.cursor()
    try:
        yield cursor
    finally:
        cursor.close()

class DuckDBSession:
    """Context manager for background tasks to get a cursor."""
    def __init__(self):
        self.con = _get_global_connection()
        
    def __enter__(self):
        self.cursor = self.con.cursor()
        return self.cursor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
