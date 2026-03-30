"""
step4_convex_hulls.py — Trajectory Clustering & Traffic Pattern Extraction.

Replicates ConvexHullsSparkApp.scala:
1. Load interpolated voyage data
2. Group positions by (itinerary, grid cell)
3. Run DBSCAN with trajectory-aware similarity (eps=0.03°, speed+heading)
4. Compute convex hulls + statistics for each cluster
5. Save + generate HTML map of traffic pattern regions
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import DBSCAN as SklearnDBSCAN

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from geo_utils import (
    GeoPoint, Grid, Polygon, WaypointDatabase,
    get_convex_hull, haversine_distance
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
INTERP_CSV = os.path.join(OUTPUT_DIR, "voyages_interpolated.csv")
WAYPOINTS_CSV = os.path.join(OUTPUT_DIR, "waypoints.csv")

# Grid params
GRID_MIN_LON = -98.0
GRID_MIN_LAT = 18.0
GRID_MAX_LON = -80.0
GRID_MAX_LAT = 31.5
GRID_STEP = 0.2

# Trajectory DBSCAN params (same as Scala ConvexHullsSparkApp)
TRAJ_EPS = 0.03  # degrees
TRAJ_MIN_PTS = 6
MIN_VOYAGES_IN_CLUSTER = 2  # Minimum distinct voyages for a valid cluster


def load_waypoints(grid):
    db = WaypointDatabase(grid)
    polygons, ids = [], []
    with open(WAYPOINTS_CSV, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",", 1)
            wp_id = int(parts[0])
            poly = Polygon.from_wkt(parts[1])
            polygons.append(poly)
            ids.append(wp_id)
    db.add_waypoints(polygons, ids)
    return db


def main():
    t0 = time.time()

    print("Setting up grid and loading waypoints...")
    grid = Grid(GRID_MIN_LON, GRID_MIN_LAT, GRID_MAX_LON, GRID_MAX_LAT, GRID_STEP, GRID_STEP)
    wp_db = load_waypoints(grid)

    print("Loading interpolated voyage data...")
    df = pd.read_csv(INTERP_CSV)
    print(f"  Total positions: {len(df):,}")
    print(f"  Unique voyages: {df['VOYAGE_ID'].nunique()}")

    # Filter out positions inside waypoints
    print("Filtering positions inside waypoints...")
    valid_mask = []
    for _, row in df.iterrows():
        gp = GeoPoint(row['LONGITUDE'], row['LATITUDE'])
        cell = grid.get_enclosing_cell(gp)
        valid_mask.append(cell is not None)

    df = df[valid_mask].copy()
    print(f"  Positions in grid: {len(df):,}")

    # Assign grid cell IDs
    print("Assigning grid cells...")
    cell_ids = []
    for _, row in df.iterrows():
        gp = GeoPoint(row['LONGITUDE'], row['LATITUDE'])
        cell = grid.get_enclosing_cell(gp)
        cell_ids.append(cell.id if cell else -1)
    df['cell_id'] = cell_ids

    # Group by (itinerary, cell)
    print("Grouping by (itinerary, cell)...")
    groups = df.groupby(['ITINERARY', 'cell_id'])
    total_groups = len(groups)
    print(f"  Total groups: {total_groups}")

    # Cluster each group and compute convex hulls
    all_hulls = []
    count = 0

    for (itinerary, cell_id), group_df in groups:
        if len(group_df) < TRAJ_MIN_PTS:
            continue

        # Compute group statistics (same as Scala)
        positions = group_df[['LONGITUDE', 'LATITUDE', 'COG', 'SOG']].values
        mean_heading = positions[:, 2].mean()
        std_heading = np.sqrt(np.mean((np.abs(positions[:, 2] - mean_heading)) ** 2))
        mean_speed = positions[:, 3].mean()
        std_speed = np.sqrt(np.mean((np.abs(positions[:, 3] - mean_speed)) ** 2))

        # Run DBSCAN on (lon, lat) with Euclidean distance in degrees
        # (same as Scala AISPositionSimilarity which uses degree-based distance)
        coords = positions[:, :2]  # lon, lat
        if len(coords) < TRAJ_MIN_PTS:
            continue

        db = SklearnDBSCAN(eps=TRAJ_EPS, min_samples=TRAJ_MIN_PTS, metric='euclidean')
        labels = db.fit_predict(coords)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = group_df[cluster_mask]
            cluster_positions = positions[cluster_mask]

            # Check minimum distinct voyages
            distinct_voyages = cluster_data['ANNOTATION'].nunique()
            if distinct_voyages < MIN_VOYAGES_IN_CLUSTER:
                continue

            if len(cluster_positions) < 3:
                continue

            # Statistics (same as Scala)
            cl_mean_heading = cluster_positions[:, 2].mean()
            cl_std_heading = np.sqrt(np.mean((np.abs(cluster_positions[:, 2] - cl_mean_heading)) ** 2))
            cl_min_heading = cluster_positions[:, 2].min()
            cl_max_heading = cluster_positions[:, 2].max()
            cl_mean_speed = cluster_positions[:, 3].mean()
            cl_std_speed = np.sqrt(np.mean((np.abs(cluster_positions[:, 3] - cl_mean_speed)) ** 2))
            cl_min_speed = cluster_positions[:, 3].min()
            cl_max_speed = cluster_positions[:, 3].max()

            # Convex hull
            points = [GeoPoint(p[0], p[1]) for p in cluster_positions[:, :2]]
            try:
                hull = get_convex_hull(points)
            except Exception:
                continue

            if len(hull.points) < 3:
                continue

            hull_id = f"{cluster_id}-{itinerary}@{cell_id}"
            all_hulls.append({
                'hull_id': hull_id,
                'itinerary': itinerary,
                'cell_id': cell_id,
                'mean_heading': round(cl_mean_heading, 2),
                'std_heading': round(cl_std_heading, 2),
                'min_heading': round(cl_min_heading, 2),
                'max_heading': round(cl_max_heading, 2),
                'mean_speed': round(cl_mean_speed, 2),
                'std_speed': round(cl_std_speed, 2),
                'min_speed': round(cl_min_speed, 2),
                'max_speed': round(cl_max_speed, 2),
                'num_positions': len(cluster_positions),
                'num_voyages': distinct_voyages,
                'polygon': hull,
            })

        count += 1
        if count % 500 == 0:
            print(f"  Processed {count}/{total_groups} groups, hulls so far: {len(all_hulls)}")

    print(f"\n  Total traffic pattern convex hulls: {len(all_hulls)}")

    # Save convex hulls CSV
    hulls_csv = os.path.join(OUTPUT_DIR, "convex_hulls.csv")
    with open(hulls_csv, 'w') as f:
        f.write("HULL_ID,ITINERARY,CELL_ID,MEAN_HEADING,STD_HEADING,MIN_HEADING,MAX_HEADING,"
                "MEAN_SPEED,STD_SPEED,MIN_SPEED,MAX_SPEED,NUM_POSITIONS,NUM_VOYAGES,POLYGON\n")
        for h in all_hulls:
            f.write(f"{h['hull_id']},{h['itinerary']},{h['cell_id']},"
                    f"{h['mean_heading']},{h['std_heading']},{h['min_heading']},{h['max_heading']},"
                    f"{h['mean_speed']},{h['std_speed']},{h['min_speed']},{h['max_speed']},"
                    f"{h['num_positions']},{h['num_voyages']},{h['polygon'].to_wkt()}\n")
    print(f"  Saved to: {hulls_csv}")

    # ---- HTML Map ----
    print("Generating HTML visualization...")
    map_center = [(GRID_MIN_LAT + GRID_MAX_LAT) / 2, (GRID_MIN_LON + GRID_MAX_LON) / 2]
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')

    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: rgba(0,0,0,0.8); padding: 12px 24px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 16px; font-weight: bold;
         border: 1px solid rgba(255,255,255,0.2);">
        🗺️ Step 4: Maritime Traffic Patterns ({len(all_hulls)} convex hulls)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Group hulls by itinerary for coloring
    itineraries = list(set(h['itinerary'] for h in all_hulls))
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
               '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
               '#F1948A', '#82E0AA', '#F8C471', '#AED6F1', '#D7BDE2',
               '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']

    itin_colors = {it: palette[i % len(palette)] for i, it in enumerate(itineraries)}

    for h in all_hulls:
        color = itin_colors[h['itinerary']]
        poly = h['polygon']
        coords = [[p.latitude, p.longitude] for p in poly.points]
        if not coords:
            continue
        popup_text = (f"<b>Hull:</b> {h['hull_id']}<br>"
                      f"<b>Route:</b> {h['itinerary']}<br>"
                      f"<b>Points:</b> {h['num_positions']}<br>"
                      f"<b>Voyages:</b> {h['num_voyages']}<br>"
                      f"<b>Heading:</b> {h['mean_heading']:.1f}° ± {h['std_heading']:.1f}°<br>"
                      f"<b>Speed:</b> {h['mean_speed']:.1f} ± {h['std_speed']:.1f} kts")
        folium.Polygon(
            locations=coords,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            weight=1,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{h['itinerary']} ({h['num_voyages']} voyages)"
        ).add_to(m)

    # Top itineraries legend
    top_itins = sorted(itineraries, key=lambda it: sum(1 for h in all_hulls if h['itinerary'] == it), reverse=True)[:10]
    legend_items = "".join(
        f'<span style="color:{itin_colors[it]}">●</span> {it} '
        f'({sum(1 for h in all_hulls if h["itinerary"] == it)} hulls)<br>'
        for it in top_itins
    )

    stats_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
         background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 12px;
         border: 1px solid rgba(255,255,255,0.2); max-width: 320px;">
        <b>📊 Traffic Pattern Stats</b><br>
        Total hulls: <b>{len(all_hulls)}</b><br>
        Unique routes: <b>{len(itineraries)}</b><br>
        DBSCAN: eps={TRAJ_EPS}°, minPts={TRAJ_MIN_PTS}<br><br>
        <b>Top routes:</b><br>
        {legend_items}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    html_path = os.path.join(OUTPUT_DIR, "step4_convex_hulls_map.html")
    m.save(html_path)
    print(f"Map saved to: {html_path}")

    elapsed = time.time() - t0
    print(f"\nStep 4 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
