"""
step2_voyages.py — Voyage Segmentation.

Replicates VoyageSparkApp.scala:
1. Load waypoint polygons from Step 1
2. For each vessel, sort positions by time
3. Segment into voyages:
   - New voyage when ship enters a different waypoint/port
   - New voyage when time gap >= 24 hours (86400s)
4. Filter: keep voyages with valid itinerary (no -1) and >= 3 positions
5. Remove positions inside waypoint polygons
6. Save + generate HTML map with sample voyages
"""

import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
import folium
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from geo_utils import GeoPoint, Grid, Polygon, WaypointDatabase

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATASET_CSV = os.path.join(OUTPUT_DIR, "dataset.csv")
WAYPOINTS_CSV = os.path.join(OUTPUT_DIR, "waypoints.csv")

# Gulf of Mexico grid  
GRID_MIN_LON = -98.0
GRID_MIN_LAT = 18.0
GRID_MAX_LON = -80.0
GRID_MAX_LAT = 31.5
GRID_STEP = 0.2  # Same as LocalDatabase default in Scala

# Voyage segmentation parameters
GAP_THRESHOLD = 86400  # 24 hours in seconds (same as Scala)
NUM_PARTITIONS = 8

# Ship type to process
SHIP_TYPE = "Cargo"  # Process Cargo first (like Scala default "Tanker")


def load_waypoints(grid):
    """Load waypoint polygons from CSV and build spatial index"""
    db = WaypointDatabase(grid)
    polygons = []
    ids = []
    with open(WAYPOINTS_CSV, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",", 1)
            wp_id = int(parts[0])
            poly = Polygon.from_wkt(parts[1])
            polygons.append(poly)
            ids.append(wp_id)
    db.add_waypoints(polygons, ids)
    print(f"  Loaded {len(polygons)} waypoint polygons")
    return db


def timestamp_to_seconds(ts_str):
    """Convert timestamp string to epoch seconds"""
    dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    return int(dt.timestamp())


def main():
    t0 = time.time()

    # ---- Create grid and load waypoints ----
    print("Setting up grid and loading waypoints...")
    grid = Grid(GRID_MIN_LON, GRID_MIN_LAT, GRID_MAX_LON, GRID_MAX_LAT, GRID_STEP, GRID_STEP)
    wp_db = load_waypoints(grid)

    # ---- Load dataset ----
    print("Loading dataset...")
    df = pd.read_csv(DATASET_CSV)
    print(f"  Total positions: {len(df):,}")

    # Process all ship types
    all_voyages = []
    ship_types_to_process = df['SHIP_TYPE'].value_counts().head(5).index.tolist()

    for ship_type in ship_types_to_process:
        print(f"\n{'='*50}")
        print(f"Segmenting voyages for: {ship_type}")
        print(f"{'='*50}")

        df_type = df[df['SHIP_TYPE'] == ship_type].copy()
        df_type['seconds'] = df_type['TIMESTAMP'].apply(timestamp_to_seconds)
        print(f"  Positions: {len(df_type):,}")

        # Group by vessel (MMSI)
        vessels = df_type.groupby('MMSI')
        v_count = 0

        for mmsi, vessel_df in vessels:
            vessel_df = vessel_df.sort_values('seconds')
            positions = vessel_df.to_dict('records')

            previous_port = -1
            voyage_id = str(uuid.uuid4())[:8]
            voyage_positions = []
            previous_seconds = positions[0]['seconds'] if positions else 0

            for pos in positions:
                gp = GeoPoint(pos['LONGITUDE'], pos['LATITUDE'])

                # Check if position is inside a waypoint
                wp_result = wp_db.get_enclosing_waypoint(gp)

                # Check time gap — split if >= 24h
                if pos['seconds'] - previous_seconds >= GAP_THRESHOLD:
                    # Save current voyage
                    if voyage_positions:
                        itinerary = f"{previous_port}_to_-1"
                        # Filter out positions inside waypoints
                        filtered = [p for p in voyage_positions
                                    if wp_db.get_enclosing_waypoint(GeoPoint(p['LONGITUDE'], p['LATITUDE'])) is None]
                        all_voyages.append({
                            'voyage_id': voyage_id,
                            'itinerary': itinerary,
                            'ship_type': ship_type,
                            'mmsi': mmsi,
                            'positions': filtered
                        })
                    voyage_positions = []
                    previous_port = -1
                    voyage_id = str(uuid.uuid4())[:8]

                if wp_result is not None:
                    port_id = wp_result[1]
                    if previous_port != port_id:
                        # New port — end this voyage, start new one
                        voyage_positions.append(pos)
                        itinerary = f"{previous_port}_to_{port_id}"
                        filtered = [p for p in voyage_positions
                                    if wp_db.get_enclosing_waypoint(GeoPoint(p['LONGITUDE'], p['LATITUDE'])) is None]
                        all_voyages.append({
                            'voyage_id': voyage_id,
                            'itinerary': itinerary,
                            'ship_type': ship_type,
                            'mmsi': mmsi,
                            'positions': filtered
                        })
                        voyage_positions = []
                        previous_port = port_id
                        voyage_id = str(uuid.uuid4())[:8]
                    else:
                        voyage_positions.append(pos)
                else:
                    voyage_positions.append(pos)

                previous_seconds = pos['seconds']

            # Save last voyage
            if voyage_positions:
                itinerary = f"{previous_port}_to_-1"
                filtered = [p for p in voyage_positions
                            if wp_db.get_enclosing_waypoint(GeoPoint(p['LONGITUDE'], p['LATITUDE'])) is None]
                all_voyages.append({
                    'voyage_id': voyage_id,
                    'itinerary': itinerary,
                    'ship_type': ship_type,
                    'mmsi': mmsi,
                    'positions': filtered
                })

            v_count += 1
            if v_count % 200 == 0:
                print(f"    Processed {v_count} vessels...")

        print(f"  Voyages before filtering: {len([v for v in all_voyages if v['ship_type'] == ship_type])}")

    # Filter: valid itinerary (no -1) and >= 3 positions (same as Scala)
    valid_voyages = [v for v in all_voyages
                     if '-1' not in v['itinerary'] and len(v['positions']) >= 3]
    print(f"\n=== Voyage filtering ===")
    print(f"  Total voyages (all types): {len(all_voyages)}")
    print(f"  Valid voyages (no -1, >= 3 positions): {len(valid_voyages)}")

    # If no valid voyages with strict port-to-port filter, relax to include all with >= 3 positions
    if len(valid_voyages) < 10:
        print("  Few valid port-to-port voyages found. Including all voyages with >= 3 positions.")
        valid_voyages = [v for v in all_voyages if len(v['positions']) >= 3]
        print(f"  Voyages with >= 3 positions: {len(valid_voyages)}")

    # ---- Save voyage data as CSV ----
    voyage_csv = os.path.join(OUTPUT_DIR, "voyages.csv")
    print(f"\nSaving {len(valid_voyages)} voyages to {voyage_csv}")
    rows = []
    for v in valid_voyages:
        for pos in v['positions']:
            rows.append({
                'VOYAGE_ID': v['voyage_id'],
                'ITINERARY': v['itinerary'],
                'MMSI': pos['MMSI'],
                'IMO': pos['IMO'],
                'LATITUDE': pos['LATITUDE'],
                'LONGITUDE': pos['LONGITUDE'],
                'COG': pos['COG'],
                'HEADING': pos['HEADING'],
                'SOG': pos['SOG'],
                'TIMESTAMP': pos['TIMESTAMP'],
                'NAME': pos['NAME'],
                'SHIP_TYPE': v['ship_type'],
                'DESTINATION': pos['DESTINATION'],
                'ANNOTATION': v['voyage_id'],
            })
    voyage_df = pd.DataFrame(rows)
    voyage_df.to_csv(voyage_csv, index=False)
    print(f"  Voyage positions saved: {len(voyage_df):,}")

    # ---- Generate HTML Map ----
    print("Generating HTML visualization...")
    map_center = [(GRID_MIN_LAT + GRID_MAX_LAT) / 2, (GRID_MIN_LON + GRID_MAX_LON) / 2]
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')

    # Title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: rgba(0,0,0,0.8); padding: 12px 24px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 16px; font-weight: bold;
         border: 1px solid rgba(255,255,255,0.2);">
        🚢 Step 2: Voyage Segmentation ({len(valid_voyages)} voyages)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Plot sample voyages (up to 100 for readability)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
              '#F1948A', '#82E0AA', '#F8C471', '#AED6F1', '#D7BDE2',
              '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']

    # Select voyages with most positions for better visualization
    display_voyages = sorted(valid_voyages, key=lambda v: len(v['positions']), reverse=True)[:100]

    for i, voyage in enumerate(display_voyages):
        color = colors[i % len(colors)]
        coords = [[p['LATITUDE'], p['LONGITUDE']] for p in voyage['positions']]
        if len(coords) < 2:
            continue
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=2,
            opacity=0.7,
            tooltip=f"Voyage {voyage['voyage_id']}<br>Type: {voyage['ship_type']}<br>"
                    f"MMSI: {voyage['mmsi']}<br>{voyage['itinerary']}<br>Pts: {len(voyage['positions'])}",
        ).add_to(m)
        # Start marker
        folium.CircleMarker(
            location=coords[0], radius=3, color='#2ECC71', fill=True,
            tooltip=f"Start: {voyage['voyage_id']}"
        ).add_to(m)
        # End marker
        folium.CircleMarker(
            location=coords[-1], radius=3, color='#E74C3C', fill=True,
            tooltip=f"End: {voyage['voyage_id']}"
        ).add_to(m)

    # Stats box
    type_counts = {}
    for v in valid_voyages:
        type_counts[v['ship_type']] = type_counts.get(v['ship_type'], 0) + 1
    type_str = "<br>".join(f"  {t}: {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))

    stats_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
         background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 13px;
         border: 1px solid rgba(255,255,255,0.2); max-width: 280px;">
        <b>📊 Voyage Statistics</b><br>
        Total voyages: <b>{len(valid_voyages)}</b><br>
        Total positions: <b>{len(voyage_df):,}</b><br>
        Showing: top {len(display_voyages)} by length<br>
        <span style="color:#2ECC71">●</span> Start
        <span style="color:#E74C3C">●</span> End<br><br>
        <b>By type:</b><br>{type_str}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    html_path = os.path.join(OUTPUT_DIR, "step2_voyages_map.html")
    m.save(html_path)
    print(f"Map saved to: {html_path}")

    elapsed = time.time() - t0
    print(f"\nStep 2 complete in {elapsed:.1f}s")
    print(f"  Total valid voyages: {len(valid_voyages)}")
    print(f"  Total positions: {len(voyage_df):,}")


if __name__ == "__main__":
    main()
