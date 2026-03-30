"""
step5_evaluation.py — Traffic Pattern Evaluation & Coverage Analysis.

Replicates IJGISApp.scala:
1. Load convex hulls from Step 4
2. Load voyage data
3. 10-fold cross-validation: test if positions fall inside training hulls
4. Compute per-cell heading/speed deviation statistics
5. Generate HTML coverage heatmap + accuracy report
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

sys.path.insert(0, os.path.dirname(__file__))
from geo_utils import GeoPoint, Grid, Polygon

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
INTERP_CSV = os.path.join(OUTPUT_DIR, "voyages_interpolated.csv")
HULLS_CSV = os.path.join(OUTPUT_DIR, "convex_hulls.csv")

# Grid params
GRID_MIN_LON = -98.0
GRID_MIN_LAT = 18.0
GRID_MAX_LON = -80.0
GRID_MAX_LAT = 31.5
GRID_STEP = 0.2


def load_hulls():
    """Load convex hulls with spatial indexing by itinerary"""
    hulls = {}  # itinerary -> list of (polygon, hull_data)
    with open(HULLS_CSV, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            hull_id = parts[0]
            itinerary = parts[1]
            # Find POLYGON in the line
            poly_start = line.find('"POLYGON')
            if poly_start == -1:
                continue
            poly_str = line[poly_start:].strip()
            try:
                poly = Polygon.from_wkt(poly_str)
            except Exception:
                continue

            if itinerary not in hulls:
                hulls[itinerary] = []
            hulls[itinerary].append({
                'hull_id': hull_id,
                'polygon': poly,
                'itinerary': itinerary,
            })
    return hulls


def point_in_any_hull(lon, lat, hull_list):
    """Check if a point falls inside any hull in the list"""
    gp = GeoPoint(lon, lat)
    for h in hull_list:
        if h['polygon'].is_inside(gp):
            return True
    return False


def main():
    t0 = time.time()

    print("Loading convex hulls...")
    hulls_by_itin = load_hulls()
    all_hulls = []
    for itin, hull_list in hulls_by_itin.items():
        all_hulls.extend(hull_list)
    print(f"  Total hulls: {len(all_hulls)}")
    print(f"  Unique itineraries: {len(hulls_by_itin)}")

    # Flatten all hull polygons for testing
    all_hull_list = all_hulls

    print("Loading voyage data...")
    df = pd.read_csv(INTERP_CSV)
    print(f"  Total positions: {len(df):,}")

    # ---- 10-fold Cross-Validation ----
    print("\n=== 10-Fold Cross-Validation ===")
    itineraries = list(hulls_by_itin.keys())
    random.seed(42)
    random.shuffle(itineraries)

    num_folds = min(10, len(itineraries))  # Handle case with few itineraries
    fold_size = len(itineraries) // num_folds if num_folds > 0 else 0

    accuracies = []
    for fold in range(num_folds):
        start = fold * fold_size
        end = start + fold_size if fold < num_folds - 1 else len(itineraries)

        test_itins = set(itineraries[start:end])
        train_itins = set(itineraries) - test_itins

        # Get train hulls
        train_hulls = []
        for itin in train_itins:
            train_hulls.extend(hulls_by_itin.get(itin, []))

        if not train_hulls:
            continue

        # Test positions (from test itineraries)
        test_df = df[df['ITINERARY'].isin(test_itins)]
        if len(test_df) == 0:
            continue

        # Sample for speed (max 5000 positions per fold)
        if len(test_df) > 5000:
            test_sample = test_df.sample(5000, random_state=fold)
        else:
            test_sample = test_df

        in_hulls = 0
        total = len(test_sample)
        for _, row in test_sample.iterrows():
            if point_in_any_hull(row['LONGITUDE'], row['LATITUDE'], train_hulls):
                in_hulls += 1

        accuracy = in_hulls / total if total > 0 else 0
        accuracies.append(accuracy)
        print(f"  Fold {fold}: accuracy={accuracy:.4f} ({in_hulls}/{total})")

    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"\n  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        mean_acc = 0
        std_acc = 0
        print("  No cross-validation results")

    # ---- Per-cell Statistics ----
    print("\n=== Computing per-cell statistics ===")
    grid = Grid(GRID_MIN_LON, GRID_MIN_LAT, GRID_MAX_LON, GRID_MAX_LAT, GRID_STEP, GRID_STEP)

    # Use vectorized cell assignment
    df['cell_lon_idx'] = ((df['LONGITUDE'] - GRID_MIN_LON) / GRID_STEP).astype(int)
    df['cell_lat_idx'] = ((df['LATITUDE'] - GRID_MIN_LAT) / GRID_STEP).astype(int)
    df['cell_key'] = df['cell_lon_idx'].astype(str) + '_' + df['cell_lat_idx'].astype(str)

    cell_stats = df.groupby('cell_key').agg(
        mean_heading=('COG', 'mean'),
        std_heading=('COG', 'std'),
        mean_speed=('SOG', 'mean'),
        std_speed=('SOG', 'std'),
        count=('COG', 'count'),
        mean_lon=('LONGITUDE', 'mean'),
        mean_lat=('LATITUDE', 'mean'),
    ).reset_index()
    cell_stats = cell_stats[cell_stats['count'] >= 5]
    print(f"  Cells with >= 5 positions: {len(cell_stats)}")

    # Save stats
    stats_csv = os.path.join(OUTPUT_DIR, "cell_statistics.csv")
    cell_stats.to_csv(stats_csv, index=False)
    print(f"  Saved to: {stats_csv}")

    # ---- HTML Map: coverage heatmap + hull visualization ----
    print("Generating HTML visualization...")
    map_center = [(GRID_MIN_LAT + GRID_MAX_LAT) / 2, (GRID_MIN_LON + GRID_MAX_LON) / 2]
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')

    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: rgba(0,0,0,0.8); padding: 12px 24px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 16px; font-weight: bold;
         border: 1px solid rgba(255,255,255,0.2);">
        📊 Step 5: Evaluation &amp; Coverage ({mean_acc:.1%} ± {std_acc:.1%} accuracy)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Feature groups
    fg_heatmap = folium.FeatureGroup(name='Traffic Density Heatmap', show=True)
    fg_hulls = folium.FeatureGroup(name='Traffic Pattern Hulls', show=True)
    fg_speed = folium.FeatureGroup(name='Speed Deviation', show=False)

    # 1. Heatmap from cell statistics
    heat_data = cell_stats[['mean_lat', 'mean_lon', 'count']].values.tolist()
    HeatMap(heat_data, radius=15, max_zoom=10, name='heatmap').add_to(fg_heatmap)

    # 2. Traffic pattern hulls  
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
               '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    for i, h in enumerate(all_hulls[:200]):  # Cap at 200 for map performance
        color = palette[i % len(palette)]
        coords = [[p.latitude, p.longitude] for p in h['polygon'].points]
        if coords:
            folium.Polygon(
                locations=coords, color=color, fill=True,
                fill_color=color, fill_opacity=0.2, weight=1,
                tooltip=f"{h['itinerary']}"
            ).add_to(fg_hulls)

    # 3. Speed deviation markers (top cells with high speed variance)
    high_speed_cells = cell_stats.nlargest(50, 'std_speed')
    for _, c in high_speed_cells.iterrows():
        folium.CircleMarker(
            location=[c['mean_lat'], c['mean_lon']],
            radius=max(3, min(c['std_speed'] * 2, 15)),
            color='#FF4444',
            fill=True,
            fill_opacity=0.5,
            tooltip=f"Speed σ={c['std_speed']:.1f}, μ={c['mean_speed']:.1f}, n={int(c['count'])}"
        ).add_to(fg_speed)

    fg_heatmap.add_to(m)
    fg_hulls.add_to(m)
    fg_speed.add_to(m)
    folium.LayerControl().add_to(m)

    # Stats box  
    fold_text = "<br>".join(f"Fold {i}: {a:.1%}" for i, a in enumerate(accuracies))
    stats_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
         background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 12px;
         border: 1px solid rgba(255,255,255,0.2); max-width: 300px; max-height: 400px; overflow-y: auto;">
        <b>📊 Evaluation Results</b><br><br>
        <b>10-Fold Cross-Validation:</b><br>
        Mean: <b style="color:#4ECDC4">{mean_acc:.1%} ± {std_acc:.1%}</b><br>
        {fold_text}<br><br>
        <b>Coverage:</b><br>
        Traffic hulls: <b>{len(all_hulls)}</b><br>
        Active cells: <b>{len(cell_stats)}</b><br>
        Total positions: <b>{len(df):,}</b><br><br>
        <b>Layers:</b><br>
        🔴 Heatmap = traffic density<br>
        🔵 Hulls = traffic patterns<br>
        🔴 Speed = speed variance hotspots
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    html_path = os.path.join(OUTPUT_DIR, "step5_evaluation_map.html")
    m.save(html_path)
    print(f"Map saved to: {html_path}")

    # Save accuracy report
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("GLASSEAS Pipeline — Gulf of Mexico Evaluation Report\n")
        f.write("=" * 55 + "\n\n")
        f.write("10-Fold Cross-Validation Results\n")
        f.write("-" * 35 + "\n")
        for i, a in enumerate(accuracies):
            f.write(f"  Fold {i}: {a:.4f}\n")
        f.write(f"\n  Mean: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"\nTraffic Patterns: {len(all_hulls)} convex hulls\n")
        f.write(f"Active Cells: {len(cell_stats)}\n")
        f.write(f"Total Positions: {len(df):,}\n")
    print(f"Report saved to: {report_path}")

    elapsed = time.time() - t0
    print(f"\nStep 5 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
