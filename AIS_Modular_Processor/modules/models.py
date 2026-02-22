import json
from datetime import datetime

class User:
    @staticmethod
    def get_by_username(con, username):
        return con.execute("SELECT * FROM users WHERE username = ?", [username]).fetchone()
        
    @staticmethod
    def get_by_id(con, user_id):
        return con.execute("SELECT * FROM users WHERE id = ?", [user_id]).fetchone()

    @staticmethod
    def get_all(con):
        res = con.execute("SELECT id, username, email, created_at FROM users").fetchall()
        return [{"id": r[0], "username": r[1], "email": r[2], "created_at": r[3]} for r in res]

    @staticmethod
    def create(con, username, email, organization=None, metadata_info=None):
        meta_str = json.dumps(metadata_info) if metadata_info else None
        con.execute(
            "INSERT INTO users (username, email) VALUES (?, ?) RETURNING id",
            [username, email]
        )
        return con.fetchone()[0]


class AnalysisRun:
    @staticmethod
    def create(con, run_name, user_id=None, config=None):
        conf_str = json.dumps(config) if config else None
        con.execute(
            "INSERT INTO analysis_runs (run_name, user_id, config, status) VALUES (?, ?, ?, 'pending') RETURNING id",
            [run_name, user_id, conf_str]
        )
        return con.fetchone()[0]

    @staticmethod
    def update_status(con, run_id, status, metadata_info=None):
        if metadata_info:
            meta_str = json.dumps(metadata_info)
            con.execute("UPDATE analysis_runs SET status = ?, metadata_info = ? WHERE id = ?", [status, meta_str, run_id])
        else:
            con.execute("UPDATE analysis_runs SET status = ? WHERE id = ?", [status, run_id])
            
        if status in ['completed', 'failed']:
            con.execute("UPDATE analysis_runs SET completed_at = current_timestamp WHERE id = ?", [run_id])

    @staticmethod
    def get(con, run_id):
        res = con.execute("SELECT id, run_name, status, config, metadata_info, started_at, completed_at FROM analysis_runs WHERE id = ?", [run_id]).fetchone()
        if not res: return None
        return {
            "id": res[0], "run_name": res[1], "status": res[2],
            "config": json.loads(res[3]) if res[3] else None,
            "metadata_info": json.loads(res[4]) if res[4] else None,
            "started_at": res[5], "completed_at": res[6]
        }

    @staticmethod
    def get_all(con):
        res = con.execute("SELECT id, run_name, status, started_at, completed_at FROM analysis_runs ORDER BY started_at DESC").fetchall()
        return [{"id": r[0], "run_name": r[1], "status": r[2], "started_at": r[3], "completed_at": r[4]} for r in res]


class SpatialRegion:
    @staticmethod
    def create(con, name, geojson_dict, user_id=None, run_id=None):
        gj_str = json.dumps(geojson_dict)
        geom = geojson_dict.get("geometry", geojson_dict)
        geom_str = json.dumps(geom)
        geom_wkt = f"ST_GeomFromGeoJSON('{geom_str}')"
        
        # DuckDB Spatial inserting GeoJSON
        query = f"""
            INSERT INTO spatial_regions (name, geojson, geom, user_id, run_id) 
            VALUES (?, ?, {geom_wkt}, ?, ?) RETURNING id
        """
        con.execute(query, [name, gj_str, user_id, run_id])
        return con.fetchone()[0]

    @staticmethod
    def get_all(con):
        res = con.execute("SELECT id, name, geojson FROM spatial_regions").fetchall()
        return [{"id": r[0], "name": r[1], "geojson": json.loads(r[2]) if r[2] else None} for r in res]

    @staticmethod
    def get(con, region_id):
        res = con.execute("SELECT id, name, geojson, run_id FROM spatial_regions WHERE id = ?", [region_id]).fetchone()
        if not res: return None
        return {"id": res[0], "name": res[1], "geojson": json.loads(res[2]) if res[2] else None, "run_id": res[3]}


class TuningResult:
    @staticmethod
    def get_by_run(con, run_id):
        res = con.execute("SELECT k_value, r_value, num_ports, silhouette_score, processing_time FROM tuning_results WHERE run_id = ?", [run_id]).fetchall()
        return [{"k": r[0], "r": r[1], "num_ports": r[2], "silhouette_score": r[3], "processing_time": r[4]} for r in res]
