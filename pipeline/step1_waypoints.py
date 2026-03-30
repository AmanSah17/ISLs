"""
step1_waypoints.py — Port/Waypoint Discovery via DBSCAN.

Replicates WaypointsApp.scala:
1. Create spatial grid for Gulf of Mexico
2. Filter stationary vessels (SOG == 0)
3. Compress: one position per (cell, MMSI) pair
4. Run DBSCAN (eps=2km, minPts=10) with haversine distance
5. Compute convex hulls for each cluster
6. Save waypoint polygons + generate HTML map
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SklearnDBSCAN
import folium

# Add pipeline to path
sys.path.insert(0, os.path.dirname(__file__))
from geo_utils import (
    GeoPoint, Grid, Cell, Polygon, get_convex_hull,
    haversine_distance, map_vessel_type
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset.csv")

# Gulf of Mexico grid parameters
GRID_MIN_LON = -98.0
GRID_MIN_LAT = 18.0
GRID_MAX_LON = -80.0
GRID_MAX_LAT = 31.5
GRID_STEP_LON = 0.01
GRID_STEP_LAT = 0.01

# Waypoint grid (coarser, for compression — same as Scala WaypointsApp)
WAYPOINT_GRID_STEP_LON = 0.01
WAYPOINT_GRID_STEP_LAT = 0.01

# DBSCAN parameters (same as Scala WaypointsApp)
EPS_METERS = 2000.0
MIN_PTS = 10


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ----
    print("Loading dataset...")
    df = pd.read_csv(DATASET_CSV)
    print(f"  Total positions: {len(df):,}")

    # Process all vessel types together (we'll run per ship type like Scala)
    # But to keep output manageable, process the top 3 types
    ship_types = df['SHIP_TYPE'].value_counts().head(5).index.tolist()
    print(f"  Processing ship types: {ship_types}")

    all_waypoints = []
    all_waypoint_ids = []
    wp_id_offset = 0

    for ship_type in ship_types:
        print(f"\n{'='*50}")
        print(f"Processing: {ship_type}")
        print(f"{'='*50}")

        df_type = df[df['SHIP_TYPE'] == ship_type].copy()
        print(f"  Positions for {ship_type}: {len(df_type):,}")

        # ---- Step 1: Filter stationary vessels (SOG == 0) ----
        # In GLASSEAS, SOG is stored * 10, so speed == 0.0 means SOG field == 0
        df_stopped = df_type[df_type['SOG'] == 0].copy()
        print(f"  Stationary positions (SOG=0): {len(df_stopped):,}")

        if len(df_stopped) < MIN_PTS:
            print(f"  Skipping {ship_type}: not enough stationary positions")
            continue

        # ---- Step 2: Compress — one position per (cell, MMSI) ----
        # Creates grid for compression (same as WaypointsApp.scala)
        print("  Creating compression grid...")
        # Instead of full grid object (expensive), use simple cell assignment
        df_stopped['cell_lon'] = ((df_stopped['LONGITUDE'] - GRID_MIN_LON) / WAYPOINT_GRID_STEP_LON).astype(int)
        df_stopped['cell_lat'] = ((df_stopped['LATITUDE'] - GRID_MIN_LAT) / WAYPOINT_GRID_STEP_LAT).astype(int)
        df_stopped['cell_key'] = df_stopped['cell_lon'].astype(str) + '_' + df_stopped['cell_lat'].astype(str)

        # Keep first occurrence per (cell, MMSI) — same deduplicate logic as Scala
        df_compressed = df_stopped.drop_duplicates(subset=['cell_key', 'MMSI'], keep='first')
        print(f"  Compressed positions: {len(df_compressed):,}")

        if len(df_compressed) < MIN_PTS:
            print(f"  Skipping {ship_type}: not enough compressed positions")
            continue

        # ---- Step 3: DBSCAN clustering ----
        print("  Running DBSCAN (eps=2km, minPts=10, haversine)...")
        coords = np.radians(df_compressed[['LATITUDE', 'LONGITUDE']].values)

        # scikit-learn DBSCAN with haversine: eps in radians = meters / earth_radius
        eps_rad = EPS_METERS / 6_371_000.0
        db = SklearnDBSCAN(eps=eps_rad, min_samples=MIN_PTS, metric='haversine', algorithm='ball_tree')
        labels = db.fit_predict(coords)

        df_compressed = df_compressed.copy()
        df_compressed['cluster'] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  Clusters found: {n_clusters}")

        if n_clusters == 0:
            print(f"  No clusters for {ship_type}")
            continue

        # ---- Step 4: Compute convex hulls ----
        print("  Computing convex hulls...")
        for cluster_id in range(n_clusters):
            cluster_pts = df_compressed[df_compressed['cluster'] == cluster_id]
            points = [GeoPoint(row['LONGITUDE'], row['LATITUDE']) for _, row in cluster_pts.iterrows()]
            if len(points) >= 3:
                hull = get_convex_hull(points)
                all_waypoints.append(hull)
                all_waypoint_ids.append(wp_id_offset + cluster_id)

        wp_id_offset += n_clusters
        print(f"  Waypoints so far: {len(all_waypoints)}")

    # ---- Save waypoints CSV ----
    waypoints_csv = os.path.join(OUTPUT_DIR, "waypoints.csv")
    print(f"\nSaving {len(all_waypoints)} waypoints to {waypoints_csv}")
    with open(waypoints_csv, 'w') as f:
        f.write("ID,POLYGON\n")
        for wp, wp_id in zip(all_waypoints, all_waypoint_ids):
            f.write(f"{wp_id},{wp.to_wkt()}\n")

    # Also save per-type files (for use by later steps)
    for ship_type in ship_types:
        safe_type = ship_type.replace("/", "_")
        per_type_csv = os.path.join(OUTPUT_DIR, f"dataset_{safe_type}_waypoints_{EPS_METERS}_{MIN_PTS}.csv")
        with open(per_type_csv, 'w') as f:
            f.write("ID,POLYGON\n")
            for wp, wp_id in zip(all_waypoints, all_waypoint_ids):
                f.write(f"{wp_id},{wp.to_wkt()}\n")

    # ---- Generate HTML Map ----
    print("Generating HTML visualization...")
    map_center = [(GRID_MIN_LAT + GRID_MAX_LAT) / 2, (GRID_MIN_LON + GRID_MAX_LON) / 2]
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')

    # Add title
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: rgba(0,0,0,0.8); padding: 12px 24px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 16px; font-weight: bold;
         border: 1px solid rgba(255,255,255,0.2);">
        🚢 Step 1: Discovered Ports &amp; Waypoints (DBSCAN eps=2km, minPts=10)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Color palette for waypoints
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
              '#F1948A', '#82E0AA', '#F8C471', '#AED6F1', '#D7BDE2']

    for i, (wp, wp_id) in enumerate(zip(all_waypoints, all_waypoint_ids)):
        color = colors[i % len(colors)]
        coords = [[p.latitude, p.longitude] for p in wp.points]
        if coords:
            folium.Polygon(
                locations=coords,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                weight=2,
                popup=f"Waypoint {wp_id}<br>Points: {len(wp.points)}",
                tooltip=f"WP-{wp_id}"
            ).add_to(m)
            # Add marker at centroid
            centroid_lat = np.mean([p.latitude for p in wp.points])
            centroid_lon = np.mean([p.longitude for p in wp.points])
            folium.CircleMarker(
                location=[centroid_lat, centroid_lon],
                radius=4,
                color=color,
                fill=True,
                popup=f"Waypoint {wp_id}"
            ).add_to(m)

    # Add stats box
    stats_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
         background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 13px;
         border: 1px solid rgba(255,255,255,0.2); max-width: 250px;">
        <b>📊 Statistics</b><br>
        Waypoints found: <b>{len(all_waypoints)}</b><br>
        Ship types: <b>{', '.join(ship_types)}</b><br>
        DBSCAN: eps=2km, minPts=10<br>
        Grid: {GRID_STEP_LON}° × {GRID_STEP_LAT}°
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    html_path = os.path.join(OUTPUT_DIR, "step1_waypoints_map.html")
    m.save(html_path)
    print(f"Map saved to: {html_path}")

    elapsed = time.time() - t0
    print(f"\nStep 1 complete in {elapsed:.1f}s")
    print(f"  Waypoints discovered: {len(all_waypoints)}")


if __name__ == "__main__":
    main()
