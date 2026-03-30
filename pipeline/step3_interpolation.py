"""
step3_interpolation.py — Voyage Trajectory Interpolation.

Replicates InterpolationSparkApp.scala:
1. Load voyage data from Step 2
2. For each voyage, use sliding window of 3 positions
3. Apply Lagrange interpolation to fill gaps (increment = 180s)
4. Calculate bearing and speed for interpolated positions
5. Save + generate HTML map comparing original vs interpolated

Memory-optimized: writes directly to CSV in chunks.
"""

import os
import sys
import time
import csv
import warnings
import numpy as np
import pandas as pd
import folium
from datetime import datetime

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(__file__))
from geo_utils import (
    GeoPoint, Grid, Polygon, WaypointDatabase,
    lagrange_interpolation, compute_bearing, haversine_distance
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
VOYAGES_CSV = os.path.join(OUTPUT_DIR, "voyages.csv")
WAYPOINTS_CSV = os.path.join(OUTPUT_DIR, "waypoints.csv")

# Gulf of Mexico grid
GRID_MIN_LON = -98.0
GRID_MIN_LAT = 18.0
GRID_MAX_LON = -80.0
GRID_MAX_LAT = 31.5
GRID_STEP = 0.2

# Interpolation parameters (same as Scala)
TIME_INCREMENT = 180  # 3 minutes in seconds
MAX_INTERP_PER_WINDOW = 200  # Cap per sliding window to prevent memory explosion
MAX_TIME_GAP = 3600  # Only interpolate gaps up to 1 hour (prevent huge fills)


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


def interpolate_voyage(positions):
    """Interpolate a single voyage — ported from InterpolationSparkApp, memory-safe"""
    if len(positions) < 3:
        return []

    # Remove duplicate coordinates
    seen_coords = set()
    unique = []
    for p in positions:
        key = (round(p['LONGITUDE'], 6), round(p['LATITUDE'], 6))
        if key not in seen_coords:
            seen_coords.add(key)
            unique.append(p)

    if len(unique) < 3:
        return []

    unique.sort(key=lambda p: p['seconds'])
    interpolated = []

    for i in range(len(unique) - 2):
        window = unique[i:i+3]
        second = window[1]
        third = window[2]

        x = np.array([w['LONGITUDE'] for w in window], dtype=np.float64)
        y = np.array([w['LATITUDE'] for w in window], dtype=np.float64)

        time_diff = abs(second['seconds'] - third['seconds'])

        # Skip very large gaps (interpolation would be inaccurate)
        if time_diff > MAX_TIME_GAP or time_diff <= TIME_INCREMENT * 2:
            continue

        num_increments = min(time_diff // TIME_INCREMENT - 1, MAX_INTERP_PER_WINDOW)
        if num_increments <= 0:
            continue

        # Longitude diff
        lon2, lon3 = second['LONGITUDE'], third['LONGITUDE']
        lon_diff = abs(lon2 - lon3)
        if lon_diff == 0:
            continue

        lon_increment = (lon3 - lon2) / (num_increments + 1)  # directional

        current_lon = second['LONGITUDE'] + lon_increment
        current_seconds = second['seconds'] + TIME_INCREMENT
        prev_lon = second['LONGITUDE']
        prev_lat = second['LATITUDE']

        for inc in range(int(num_increments)):
            try:
                # Check for duplicate x values
                if len(set(x)) < len(x):
                    break
                interp_lat = lagrange_interpolation(x, y, current_lon)
            except (ZeroDivisionError, FloatingPointError, OverflowError, ValueError):
                break

            # Bounds check
            if not (-90 <= interp_lat <= 90) or not (-180 <= current_lon <= 180):
                current_lon += lon_increment
                current_seconds += TIME_INCREMENT
                continue

            # Speed check
            try:
                dist = haversine_distance(GeoPoint(prev_lon, prev_lat), GeoPoint(current_lon, interp_lat))
            except (ValueError, OverflowError):
                break
            speed_knots = dist / 0.05 / 1.852
            if speed_knots > 50:
                current_lon += lon_increment
                current_seconds += TIME_INCREMENT
                continue

            bearing = compute_bearing(prev_lon, prev_lat, current_lon, interp_lat)
            dt = datetime.fromtimestamp(current_seconds)
            formatted = dt.strftime('%Y-%m-%d %H:%M:%S')

            interpolated.append({
                'VOYAGE_ID': second['VOYAGE_ID'],
                'ITINERARY': second['ITINERARY'],
                'MMSI': second['MMSI'],
                'IMO': second['IMO'],
                'LATITUDE': round(interp_lat, 6),
                'LONGITUDE': round(current_lon, 6),
                'COG': round(bearing, 2),
                'HEADING': round(bearing, 2),
                'SOG': round(speed_knots * 10) / 10.0,
                'TIMESTAMP': formatted,
                'NAME': second['NAME'],
                'SHIP_TYPE': second['SHIP_TYPE'],
                'DESTINATION': second['DESTINATION'],
                'ANNOTATION': second['ANNOTATION'],
                'is_interpolated': True,
                'seconds': current_seconds,
            })

            prev_lon = current_lon
            prev_lat = interp_lat
            current_lon += lon_increment
            current_seconds += TIME_INCREMENT

    return interpolated


def main():
    t0 = time.time()

    print("Setting up grid and loading waypoints...")
    grid = Grid(GRID_MIN_LON, GRID_MIN_LAT, GRID_MAX_LON, GRID_MAX_LAT, GRID_STEP, GRID_STEP)
    wp_db = load_waypoints(grid)

    print("Loading voyage data...")
    df = pd.read_csv(VOYAGES_CSV)
    df['seconds'] = df['TIMESTAMP'].apply(lambda ts: int(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()))
    print(f"  Total voyage positions: {len(df):,}")
    print(f"  Unique voyages: {df['VOYAGE_ID'].nunique()}")

    # Output CSV (stream write)
    interp_csv = os.path.join(OUTPUT_DIR, "voyages_interpolated.csv")
    columns = ['VOYAGE_ID', 'ITINERARY', 'MMSI', 'IMO', 'LATITUDE', 'LONGITUDE',
               'COG', 'HEADING', 'SOG', 'TIMESTAMP', 'NAME', 'SHIP_TYPE',
               'DESTINATION', 'ANNOTATION', 'is_interpolated']

    total_original = 0
    total_interpolated = 0
    sample_voyages = {}  # Store a few for visualization

    with open(interp_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        count = 0
        for voyage_id, voyage_df in df.groupby('VOYAGE_ID'):
            positions = voyage_df.to_dict('records')

            # Write original positions
            for p in positions:
                row = {k: p.get(k, '') for k in columns}
                row['is_interpolated'] = False
                writer.writerow(row)
            total_original += len(positions)

            # Interpolate and write
            new_pts = interpolate_voyage(positions)
            for p in new_pts:
                row = {k: p.get(k, '') for k in columns}
                writer.writerow(row)
            total_interpolated += len(new_pts)

            # Store samples for visualization
            if len(new_pts) > 5 and len(sample_voyages) < 30:
                sample_voyages[voyage_id] = {
                    'original': positions,
                    'interpolated': new_pts
                }

            count += 1
            if count % 1000 == 0:
                print(f"  Processed {count} voyages... (orig: {total_original:,}, interp: {total_interpolated:,})")

    print(f"\n  Original positions: {total_original:,}")
    print(f"  New interpolated: {total_interpolated:,}")
    print(f"  Total: {total_original + total_interpolated:,}")
    print(f"  Saved to: {interp_csv}")

    # ---- HTML Map ----
    print("Generating HTML visualization...")
    map_center = [(GRID_MIN_LAT + GRID_MAX_LAT) / 2, (GRID_MIN_LON + GRID_MAX_LON) / 2]
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')

    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: rgba(0,0,0,0.8); padding: 12px 24px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 16px; font-weight: bold;
         border: 1px solid rgba(255,255,255,0.2);">
        🧭 Step 3: Lagrange Interpolation (Δt=3min, max gap=1hr)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    fg_original = folium.FeatureGroup(name='Original Trajectories', show=True)
    fg_interpolated = folium.FeatureGroup(name='Interpolated Trajectories', show=True)

    colors_o = ['#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22',
                '#3498DB', '#2ECC71', '#D35400', '#8E44AD', '#16A085']
    colors_i = ['#FF9999', '#FFCC66', '#CC99FF', '#66FFCC', '#FFAA66',
                '#66CCFF', '#99FF99', '#FF8833', '#BB77DD', '#33DDBB']

    for idx, (vid, data) in enumerate(sample_voyages.items()):
        co = colors_o[idx % len(colors_o)]
        ci = colors_i[idx % len(colors_i)]

        orig = sorted(data['original'], key=lambda p: p.get('seconds', 0))
        if len(orig) >= 2:
            coords = [[p['LATITUDE'], p['LONGITUDE']] for p in orig]
            folium.PolyLine(coords, color=co, weight=3, opacity=0.9,
                            tooltip=f"Original: {vid} ({len(orig)} pts)").add_to(fg_original)

        # Full trajectory (original + interpolated)
        all_pts = sorted(data['original'] + data['interpolated'], key=lambda p: p.get('seconds', 0))
        if len(all_pts) >= 2:
            coords = [[p['LATITUDE'], p['LONGITUDE']] for p in all_pts]
            folium.PolyLine(coords, color=ci, weight=1.5, opacity=0.7, dash_array='5 5',
                            tooltip=f"Interp: {vid} (+{len(data['interpolated'])} pts)").add_to(fg_interpolated)

    fg_original.add_to(m)
    fg_interpolated.add_to(m)
    folium.LayerControl().add_to(m)

    stats_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
         background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 13px;
         border: 1px solid rgba(255,255,255,0.2); max-width: 280px;">
        <b>📊 Interpolation Stats</b><br>
        Original: <b>{total_original:,}</b><br>
        New interpolated: <b>{total_interpolated:,}</b><br>
        Total: <b>{total_original + total_interpolated:,}</b><br>
        Time step: <b>180s</b> | Max gap: <b>1hr</b><br>
        Method: <b>Lagrange (3-point)</b><br><br>
        <span style="color:#E74C3C">━━</span> Original &nbsp;
        <span style="color:#FF9999">╌╌</span> Interpolated
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    html_path = os.path.join(OUTPUT_DIR, "step3_interpolation_map.html")
    m.save(html_path)
    print(f"Map saved to: {html_path}")

    elapsed = time.time() - t0
    print(f"\nStep 3 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
