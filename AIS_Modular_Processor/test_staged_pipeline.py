import requests
import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestStagedPipeline")

API_BASE = "http://127.0.0.1:8000"

# Assuming you already have 'consolidated_ais_data_20200105_20200112' uploaded
# Check API for regions or create one
logger.info("Setting up region...")
bbox = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-118.4, 33.6],
            [-118.1, 33.6],
            [-118.1, 33.9],
            [-118.4, 33.9],
            [-118.4, 33.6]
        ]]
    },
    "properties": {"name": "Test Staged Region"}
}

res = requests.post(f"{API_BASE}/regions?name=StagedTestRegion", json=bbox)
region_id = res.json()['id']
logger.info(f"Region created: {region_id}")

config = {
    "plsn": {"k": 300, "r": 0.0001},
    "nlsn": {"gamma": 0.05},
    "tuning": None
}

def wait_for_status(task_id, target_status):
    while True:
        res = requests.get(f"{API_BASE}/status/{task_id}").json()
        logger.info(f"Task {task_id} Status: {res['status']}")
        if res['status'] == target_status:
            return res
        if res['status'] == 'failed':
            raise Exception(f"Task failed: {res.get('metadata')}")
        time.sleep(1.5)

# 1. Start PLSN
logger.info("\n=== STARTING PLSN ===")
res = requests.post(f"{API_BASE}/analyze/plsn", json={
    "ais_file_id": "0e30f4f7",
    "region_id": region_id,
    "run_name": "Staged_Test_Run",
    "config": config
})
data = res.json()
print("SERVER RESPONSE:", data)
task_id = data['task_id']
run_id = data['run_id']
logger.info(f"PLSN Task started: {task_id} for Run {run_id}")

final_res = wait_for_status(task_id, "completed_plsn")
logger.info(f"PLSN Completed. Found {final_res['metadata'].get('plsn_nodes', 0)} nodes.\n")

# 2. Start NLSN
logger.info("=== STARTING NLSN ===")
res = requests.post(f"{API_BASE}/analyze/nlsn", json={
    "run_id": run_id,
    "config": config
})
data = res.json()
task_id = data['task_id']
logger.info(f"NLSN Task started: {task_id}")

final_res = wait_for_status(task_id, "completed_nlsn")
logger.info(f"NLSN Completed. Edges processed.\n")

# 3. Start RLSN
logger.info("=== STARTING RLSN ===")
res = requests.post(f"{API_BASE}/analyze/rlsn", json={
    "run_id": run_id,
    "config": config
})
data = res.json()
task_id = data['task_id']
logger.info(f"RLSN Task started: {task_id}")

final_res = wait_for_status(task_id, "completed")
logger.info(f"RLSN Completed. Routes processed: {final_res['metadata'].get('rlsn_routes', 0)}\n")

logger.info("PIPELINE TEST SUCCESSFUL.")
