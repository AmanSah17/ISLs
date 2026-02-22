from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import logging
import shutil
import json
from datetime import datetime
import duckdb

from modules.data_loader import AISDataLoader
from modules.extraction import ExtractionOrchestrator
from modules.db_session import get_db, DuckDBSession
from modules.models import User, AnalysisRun, SpatialRegion, TuningResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIS_API")

app = FastAPI(title="AIS Modular Analysis API")

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs/api_runs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory fast cache for Task Status
TASK_STATUS_CACHE = {}

@app.on_event("shutdown")
def shutdown_event():
    from modules.db_session import _global_con
    if _global_con is not None:
        logger.info("Closing global DuckDB connection...")
        _global_con.close()

# =============================================================================
# Pydantic Schemas
# =============================================================================
class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None
    organization: Optional[str] = None
    metadata_info: Optional[dict] = {}

class AnalysisRequest(BaseModel):
    ais_file_id: Optional[str] = None
    region_id: Optional[int] = None
    user_id: Optional[int] = None
    run_name: Optional[str] = None
    run_id: Optional[int] = None
    config: dict = {
        "plsn": {"k": 1400, "r": 0.0001},
        "nlsn": {"gamma": 0.05},
        "tuning": None
    }

# =============================================================================
# User Endpoints
# =============================================================================
@app.post("/users")
async def create_user(u: UserCreate, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    if User.get_by_username(db, u.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    uid = User.create(db, u.username, u.email, u.organization, u.metadata_info)
    return {"id": uid, "username": u.username, "created_at": datetime.utcnow().isoformat()}

@app.get("/users")
async def list_users(db: duckdb.DuckDBPyConnection = Depends(get_db)):
    return User.get_all(db)

@app.get("/users/{user_id}")
async def get_user_route(user_id: int, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    u = User.get_by_id(db, user_id)
    if not u: raise HTTPException(status_code=404, detail="User not found")
    return {"id": u[0], "username": u[1], "email": u[2], "created_at": u[3], "is_active": u[4]}

# =============================================================================
# File Upload
# =============================================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".parquet", ".csv"]:
        raise HTTPException(status_code=400, detail="Only .parquet or .csv files supported")
    
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"file_id": file_id, "filename": file.filename, "size_bytes": os.path.getsize(file_path)}

# =============================================================================
# Region Endpoints
# =============================================================================
@app.post("/regions")
async def save_region(name: str, geojson: dict, description: str = "", user_id: int = None, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    try:
        rid = SpatialRegion.create(db, name, geojson, user_id, None)
        return {"id": rid, "name": name, "created_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving region: {e}")

@app.get("/regions")
async def list_regions(user_id: int = None, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    # Note: simple filtering by user_id not in helper yet but all is fine
    return SpatialRegion.get_all(db)

# =============================================================================
# Analysis Endpoints
# =============================================================================
@app.post("/analyze/plsn")
async def start_plsn(request: AnalysisRequest, background_tasks: BackgroundTasks, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    if not request.ais_file_id or not request.region_id:
        raise HTTPException(status_code=400, detail="ais_file_id and region_id are required for PLSN")

    task_id = str(uuid.uuid4())
    run_name = request.run_name or f"run_{task_id[:8]}"

    ais_path = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.startswith(request.ais_file_id)), None)
    if not ais_path:
        raise HTTPException(status_code=404, detail="AIS file not found. Please upload first.")

    region = SpatialRegion.get(db, request.region_id)
    if not region:
        raise HTTPException(status_code=404, detail="Region not found.")

    run_id = AnalysisRun.create(db, run_name, request.user_id, request.config)
    
    # Update Region to link to this run
    db.execute("UPDATE spatial_regions SET run_id = ? WHERE id = ?", [run_id, request.region_id])

    TASK_STATUS_CACHE[task_id] = {"run_id": run_id, "status": "pending", "metadata": {}}
    background_tasks.add_task(run_plsn_task, task_id, run_id, ais_path, region["geojson"], request.config)
    return {"task_id": task_id, "run_id": run_id, "run_name": run_name}

@app.post("/analyze/nlsn")
async def start_nlsn(request: AnalysisRequest, background_tasks: BackgroundTasks, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    if not request.run_id:
        raise HTTPException(status_code=400, detail="run_id is required for NLSN generation")
        
    task_id = str(uuid.uuid4())
    TASK_STATUS_CACHE[task_id] = {"run_id": request.run_id, "status": "pending", "metadata": {}}
    background_tasks.add_task(run_nlsn_task, task_id, request.run_id, request.config)
    return {"task_id": task_id, "run_id": request.run_id}

@app.post("/analyze/rlsn")
async def start_rlsn(request: AnalysisRequest, background_tasks: BackgroundTasks, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    if not request.run_id:
        raise HTTPException(status_code=400, detail="run_id is required for RLSN generation")
        
    task_id = str(uuid.uuid4())
    TASK_STATUS_CACHE[task_id] = {"run_id": request.run_id, "status": "pending", "metadata": {}}
    background_tasks.add_task(run_rlsn_task, task_id, request.run_id, request.config)
    return {"task_id": task_id, "run_id": request.run_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    if task_id not in TASK_STATUS_CACHE:
        # Fallback to DB
        res = db.execute("SELECT run_id FROM spatial_regions LIMIT 1") # fallback minimal lookup
        raise HTTPException(status_code=404, detail="Task not found in cache")
    return {"task_id": task_id, **TASK_STATUS_CACHE[task_id]}

@app.get("/runs")
async def list_runs(user_id: int = None, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    return AnalysisRun.get_all(db)

@app.get("/runs/{run_id}")
async def get_run_detail(run_id: int, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    run = AnalysisRun.get(db, run_id)
    if not run: raise HTTPException(status_code=404, detail="Run not found")
    
    tuning = TuningResult.get_by_run(db, run_id)
    
    # Count metrics dynamically from DuckDB
    nodes = db.execute("SELECT COUNT(*) FROM port_nodes WHERE run_id = ?", [run_id]).fetchone()[0]
    traj = db.execute("SELECT COUNT(*) FROM vessel_trajectories WHERE run_id = ?", [run_id]).fetchone()[0]
    
    return {
        "id": run["id"], "run_name": run["run_name"], "status": run["status"],
        "config": run["config"], "created_at": run["started_at"],
        "tuning_results": tuning,
        "node_count": nodes,
        "trajectory_count": traj
    }

@app.get("/runs/{run_id}/network")
async def get_integrated_network(run_id: int, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    """Extracts all layers (PLSN, NLSN, RLSN) for a given run and returns a merged GeoJSON FeatureCollection."""
    features = []
    
    # 1. Fetch PLSN Nodes (Ports)
    nodes = db.execute("SELECT id, ST_AsGeoJSON(geom) FROM port_nodes WHERE run_id = ?", [run_id]).fetchall()
    for row in nodes:
        if row[1]:
            gj = json.loads(row[1])
            features.append({
                "type": "Feature",
                "geometry": gj,
                "properties": {"layer": "PLSN", "type": "port_node", "db_id": row[0]}
            })
            
    # 2. Fetch NLSN Edges (Shipping Lanes)
    edges = db.execute("SELECT id, ST_AsGeoJSON(geom) FROM shipping_edges WHERE run_id = ?", [run_id]).fetchall()
    for row in edges:
        if row[1]:
            gj = json.loads(row[1])
            features.append({
                "type": "Feature",
                "geometry": gj,
                "properties": {"layer": "NLSN", "type": "shipping_edge", "db_id": row[0]}
            })
            
    # 3. Fetch RLSN Trajectories (Raw Routes)
    trajs = db.execute("SELECT mmsi, ST_AsGeoJSON(geom) FROM vessel_trajectories WHERE run_id = ?", [run_id]).fetchall()
    for row in trajs:
        if row[1]:
            gj = json.loads(row[1])
            features.append({
                "type": "Feature",
                "geometry": gj,
                "properties": {"layer": "RLSN", "type": "trajectory", "mmsi": row[0]}
            })
            
    return {"type": "FeatureCollection", "features": features}
import pandas as pd

def run_plsn_task(task_id, run_id, ais_path, region_gj, config):
    output_path = os.path.join(OUTPUT_DIR, f"run_{run_id}")
    os.makedirs(output_path, exist_ok=True)
    
    with DuckDBSession() as db:
        try:
            AnalysisRun.update_status(db, run_id, "processing")
            TASK_STATUS_CACHE[task_id]["status"] = "processing"

            reg_file = os.path.join(output_path, "region.geojson")
            with open(reg_file, "w") as f:
                json.dump(region_gj, f)

            logger.info(f"[Run {run_id}] Loading/clipping via DuckDB from {ais_path}")
            loader = AISDataLoader()
            clipped_df = loader.clip_by_region(ais_path, reg_file)

            if clipped_df.empty:
                err = "No AIS data within the selected region."
                AnalysisRun.update_status(db, run_id, "failed", {"error": err})
                TASK_STATUS_CACHE[task_id]["status"] = "failed"
                TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update({"error": err})
                return

            # Save state for future stages
            clipped_df.to_parquet(os.path.join(output_path, "clipped_ais.parquet"))

            orchestrator = ExtractionOrchestrator(output_path, run_id, task_id, TASK_STATUS_CACHE)
            results = orchestrator.run_plsn(clipped_df, config)

            # Save PLSN results to disk for next stage
            with open(os.path.join(output_path, "plsn_results.json"), "w") as f:
                json.dump(results, f)

            # Persist Tuning
            if results.get('tuning'):
                for t in results['tuning']:
                    db.execute(
                        "INSERT INTO tuning_results (run_id, k_value, r_value, num_ports, silhouette_score) VALUES (?, ?, ?, ?, ?)",
                        [run_id, t['k'], t['r'], t['clusters_found'], t['noise_points']]
                    )

            # Route Edges
            for edge in results['plsn'].get('edges', []):
                s_lat, s_lon = edge.get('src_lat'), edge.get('src_lon')
                d_lat, d_lon = edge.get('dst_lat'), edge.get('dst_lon')
                if all([s_lat, s_lon, d_lat, d_lon]):
                    geom = f"ST_GeomFromText('LINESTRING({s_lon} {s_lat}, {d_lon} {d_lat})')"
                    db.execute(f"INSERT INTO shipping_edges (run_id, geom) VALUES (?, {geom})", [run_id])

            # Persist Nodes
            for node in results['plsn'].get('nodes', []):
                lat, lon = node.get('lat'), node.get('lon')
                if lat is not None and lon is not None:
                    geom = f"ST_GeomFromText('POINT({lon} {lat})')"
                    db.execute(
                        f"INSERT INTO port_nodes (run_id, cluster_id, geom) VALUES (?, ?, {geom})",
                        [run_id, node.get('port_id')]
                    )

            meta = {"plsn_nodes": len(results['plsn'].get('nodes', []))}
            AnalysisRun.update_status(db, run_id, "completed_plsn", meta)
            TASK_STATUS_CACHE[task_id]["status"] = "completed"
            TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update(meta)
            logger.info(f"[Run {run_id}] PLSN COMPLETED.")

        except Exception as e:
            logger.error(f"[Run {run_id}] PLSN failed: {e}", exc_info=True)
            AnalysisRun.update_status(db, run_id, "failed", {"error": str(e)})
            TASK_STATUS_CACHE[task_id]["status"] = "failed"
            TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update({"error": str(e)})

def run_nlsn_task(task_id, run_id, config):
    output_path = os.path.join(OUTPUT_DIR, f"run_{run_id}")
    
    with DuckDBSession() as db:
        try:
            TASK_STATUS_CACHE[task_id]["status"] = "processing"

            # Load State
            import pandas as pd
            import json
            clipped_df = pd.read_parquet(os.path.join(output_path, "clipped_ais.parquet"))
            with open(os.path.join(output_path, "plsn_results.json"), "r") as f:
                plsn_state = json.load(f)["plsn"]

            orchestrator = ExtractionOrchestrator(output_path, run_id, task_id, TASK_STATUS_CACHE)
            results = orchestrator.run_nlsn(clipped_df, plsn_state, config)

            # Save NLSN state
            with open(os.path.join(output_path, "nlsn_results.json"), "w") as f:
                json.dump(results, f)

            meta = {"nlsn_edges": len(results['nlsn'].get('edges', []))}
            AnalysisRun.update_status(db, run_id, "completed_nlsn", meta)
            TASK_STATUS_CACHE[task_id]["status"] = "completed"
            TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update(meta)
            logger.info(f"[Run {run_id}] NLSN COMPLETED.")

        except Exception as e:
            logger.error(f"[Run {run_id}] NLSN failed: {e}", exc_info=True)
            AnalysisRun.update_status(db, run_id, "failed", {"error": str(e)})
            TASK_STATUS_CACHE[task_id]["status"] = "failed"
            TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update({"error": str(e)})

def run_rlsn_task(task_id, run_id, config):
    output_path = os.path.join(OUTPUT_DIR, f"run_{run_id}")
    
    with DuckDBSession() as db:
        try:
            TASK_STATUS_CACHE[task_id]["status"] = "processing"

            # Load State
            import pandas as pd
            import json
            clipped_df = pd.read_parquet(os.path.join(output_path, "clipped_ais.parquet"))
            with open(os.path.join(output_path, "plsn_results.json"), "r") as f:
                plsn_state = json.load(f)["plsn"]

            orchestrator = ExtractionOrchestrator(output_path, run_id, task_id, TASK_STATUS_CACHE)
            results = orchestrator.run_rlsn(clipped_df, plsn_state, config)

            # Trajectories
            for feat in results['rlsn'].get('routes', {}).get('features', []):
                coords = feat.get('geometry', {}).get('coordinates', [])
                if len(coords) >= 2:
                    wkt_line = f"LINESTRING({', '.join([f'{c[0]} {c[1]}' for c in coords])})"
                    geom = f"ST_GeomFromText('{wkt_line}')"
                    mmsi = feat['properties'].get('mmsi')
                    db.execute(f"INSERT INTO vessel_trajectories (run_id, mmsi, geom) VALUES (?, ?, {geom})", [run_id, str(mmsi)])

            meta = {"rlsn_routes": len(results['rlsn'].get('routes', {}).get('features', []))}
            AnalysisRun.update_status(db, run_id, "completed", meta)  # Final complete state
            TASK_STATUS_CACHE[task_id]["status"] = "completed"
            TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update(meta)
            logger.info(f"[Run {run_id}] RLSN COMPLETED.")

        except Exception as e:
            logger.error(f"[Run {run_id}] RLSN failed: {e}", exc_info=True)
            AnalysisRun.update_status(db, run_id, "failed", {"error": str(e)})
            TASK_STATUS_CACHE[task_id]["status"] = "failed"
            TASK_STATUS_CACHE[task_id].setdefault("metadata", {}).update({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
