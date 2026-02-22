"""
Standalone Orchestrator for Route-Level Shipping Network (RLSN) Extraction.
Loads NLSN waypoints and AIS trajectories to derive customary routes and boundaries.
"""

import logging
import os
import time
import pandas as pd
import json

from config import (
    AIS_SOURCE, NLSN_ARTIFACTS_DIR, RESULTS_DIR,
    RLSN_NUM_SLICES, RLSN_SEARCH_RADIUS, RLSN_MAX_POINTS, RLSN_SIGMA_MULT,
    MAP_CENTER, MAP_ZOOM
)
from modules.rlsn_generator import RLSNGenerator
from modules.visualizer import RLSNVisualizer

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RLSN_Pipeline")

def main():
    logger.info("Initializing Standalone RLSN Pipeline...")
    
    # 1. Verification of inputs
    if not os.path.exists(AIS_SOURCE):
        logger.error(f"AIS Source not found: {AIS_SOURCE}")
        return
    
    if not os.path.exists(NLSN_ARTIFACTS_DIR):
        logger.error(f"NLSN Artifacts directory not found: {NLSN_ARTIFACTS_DIR}")
        return

    # 2. Setup output versioning
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Results will be saved to: {output_path}")

    # 3. Load Data
    logger.info("Loading NLSN waypoints and edges...")
    nodes_df = pd.read_csv(os.path.join(NLSN_ARTIFACTS_DIR, "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(NLSN_ARTIFACTS_DIR, "edges.csv"))
    
    logger.info(f"Loading AIS data (Parquet) from {AIS_SOURCE}...")
    ais_full = pd.read_parquet(AIS_SOURCE, columns=["LAT", "LON"])
    if len(ais_full) > RLSN_MAX_POINTS:
        logger.info(f"Sampling {RLSN_MAX_POINTS} points for performance...")
        ais_df = ais_full.sample(n=RLSN_MAX_POINTS, random_state=42)
    else:
        ais_df = ais_full

    # 4. Run RLSN Extraction
    logger.info("Running Gaussian traffic flow fitting...")
    generator = RLSNGenerator(
        output_dir=output_path,
        num_slices=RLSN_NUM_SLICES,
        search_radius=RLSN_SEARCH_RADIUS,
        sigma_mult=RLSN_SIGMA_MULT
    )
    
    start_time = time.time()
    routes, bounds = generator.extract_rlsn(ais_df, nodes_df, edges_df)
    elapsed = time.time() - start_time
    logger.info(f"Extraction complete in {elapsed:.2f} seconds.")

    # 5. Generate Visualization
    logger.info("Generating interactive map...")
    viz_path = os.path.join(output_path, "rlsn_map_standalone.html")
    visualizer = RLSNVisualizer(output_file=viz_path, center=MAP_CENTER, zoom=MAP_ZOOM)
    
    # Bundle GeoJSON for the visualizer
    routes_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"source": r["source"], "target": r["target"]},
            "geometry": r["geometry"]
        } for r in routes]
    }
    bounds_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"source": b["source"], "target": b["target"]},
            "geometry": b["geometry"]
        } for b in bounds]
    }
    
    # Sample heatmap points for background
    heatmap_points = ais_df.sample(min(100000, len(ais_df))).values.tolist()
    
    visualizer.generate_map(routes_geojson, bounds_geojson, heatmap_points)
    
    logger.info("Pipeline completed successfully!")
    print(f"\n[SUCCESS] Standalone RLSN results generated.")
    print(f"Map: {viz_path}")
    print(f"Routes: {os.path.join(output_path, 'rlsn_routes.geojson')}")
    print(f"Boundaries: {os.path.join(output_path, 'rlsn_boundaries.geojson')}")

if __name__ == "__main__":
    main()
