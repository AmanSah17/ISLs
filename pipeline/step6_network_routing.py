"""
step6_network_routing.py — Contiguous Route Generation (Per Vessel Type)

1. Load Waypoints (start/end anchors).
2. Load Convex Hulls (from Step 4). Calculate their centroids.
3. For each top itinerary, chain the hull centroids starting from the origin port using Nearest Neighbor matching.
4. Load Voyages to associate a vessel type with each itinerary.
5. Create an interactive map showing continuous polylines.
   - Lines are color-coded and layered by vessel type.
   - Line thickness corresponds to route volume.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import folium
from math import radians, cos, sin, asin, sqrt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
HULLS_CSV = os.path.join(OUTPUT_DIR, "convex_hulls.csv")
WAYPOINTS_CSV = os.path.join(OUTPUT_DIR, "waypoints.csv")
VOYAGES_CSV = os.path.join(OUTPUT_DIR, "voyages.csv")

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def parse_polygon_wkt(wkt_str):
    """Extracts longitudes and latitudes from WKT string 'POLYGON ((lon lat, ...))'"""
    inner = wkt_str.replace("POLYGON", "").replace("(", "").replace(")", "").strip()
    c_pairs = inner.split(",")
    lons, lats = [], []
    for pair in c_pairs:
        lon, lat = pair.strip().split()
        lons.append(float(lon))
        lats.append(float(lat))
    return lons, lats

def load_waypoints_centroids():
    wp_centroids = {}
    if not os.path.exists(WAYPOINTS_CSV):
        return wp_centroids
        
    with open(WAYPOINTS_CSV, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",", 1)
            wp_id = parts[0]
            poly_str = parts[1].strip('"').strip()
            if poly_str.startswith("POLYGON"):
                lons, lats = parse_polygon_wkt(poly_str)
                if lats and lons:
                    wp_centroids[wp_id] = (sum(lons)/len(lons), sum(lats)/len(lats))
    return wp_centroids

def main():
    t0 = time.time()
    
    print("Loading waypoint centroids...")
    wp_centroids = load_waypoints_centroids()
    
    print("Loading convex hulls...")
    df_hulls = pd.read_csv(HULLS_CSV)
    
    # Calculate centroids for all hulls
    centroids_lon = []
    centroids_lat = []
    
    for _, row in df_hulls.iterrows():
        poly_str = row['POLYGON']
        try:
            lons, lats = parse_polygon_wkt(poly_str)
            centroids_lon.append(sum(lons)/len(lons))
            centroids_lat.append(sum(lats)/len(lats))
        except:
            centroids_lon.append(None)
            centroids_lat.append(None)
            
    df_hulls['centroid_lon'] = centroids_lon
    df_hulls['centroid_lat'] = centroids_lat
    df_hulls = df_hulls.dropna(subset=['centroid_lon', 'centroid_lat'])
    
    print("Loading voyage data for vessel type mapping...")
    df_voyages = pd.read_csv(VOYAGES_CSV)
    
    # Find dominant ship type per itinerary
    itin_ship_types = df_voyages.groupby('ITINERARY')['SHIP_TYPE'].agg(
        lambda x: x.value_counts().index[0] if not x.empty else 'Unknown'
    ).to_dict()
    
    # Process top itineraries
    itinerary_counts = df_hulls['ITINERARY'].value_counts()
    print(f"\nTotal itineraries: {len(itinerary_counts)}")
    
    # We will draw routes that have at least 2 hulls for a contiguous line
    valid_itineraries = itinerary_counts[itinerary_counts >= 2].index.tolist()
    print(f"Itineraries with >= 2 hulls (drawable routes): {len(valid_itineraries)}")
    
    routes = []
    
    for itin in valid_itineraries:
        itin_hulls = df_hulls[df_hulls['ITINERARY'] == itin].copy()
        
        parts = itin.split('_to_')
        start_wp = parts[0]
        end_wp = parts[1] if len(parts) > 1 else '-1'
        
        # Determine sequence
        points = []
        
        # Add start waypoint if exists
        start_added = False
        if start_wp in wp_centroids:
            points.append({
                'lon': wp_centroids[start_wp][0], 
                'lat': wp_centroids[start_wp][1],
                'is_port': True,
                'id': f"Port_{start_wp}"
            })
            start_added = True
            
        # Get all hull centroids for this itinerary
        hull_points = []
        for _, row in itin_hulls.iterrows():
            hull_points.append({
                'lon': row['centroid_lon'],
                'lat': row['centroid_lat'],
                'is_port': False,
                'id': row['HULL_ID'],
                'voyages': row['NUM_VOYAGES'] 
            })
            
        # Nearest Neighbor Sequencing
        if start_added and hull_points:
            # Start from the port
            current_pt = points[0]
            unvisited = hull_points.copy()
            
            while unvisited:
                # Find nearest unvisited hull
                nearest_idx = -1
                min_dist = float('inf')
                
                for i, cand in enumerate(unvisited):
                    dist = haversine(current_pt['lon'], current_pt['lat'], cand['lon'], cand['lat'])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = i
                        
                # Add nearest to sequence
                next_pt = unvisited.pop(nearest_idx)
                points.append(next_pt)
                current_pt = next_pt
        else:
            # If no start port, just sort geographically (e.g., by longitude) as a fallback
            hull_points.sort(key=lambda x: x['lon'])
            points.extend(hull_points)
            
        # Add end waypoint if exists
        if end_wp in wp_centroids:
            points.append({
                'lon': wp_centroids[end_wp][0], 
                'lat': wp_centroids[end_wp][1],
                'is_port': True,
                'id': f"Port_{end_wp}"
            })
            
        if len(points) >= 2:
            # Number of voyages is roughly the median of the hulls forming it
            avg_voyages = int(itin_hulls['NUM_VOYAGES'].median())
            
            routes.append({
                'itinerary': itin,
                'ship_type': itin_ship_types.get(itin, 'Unknown'),
                'num_hulls': len(itin_hulls),
                'avg_voyages': avg_voyages,
                'points': points
            })

    print(f"Generated {len(routes)} continuous route networks.")
    
    # ---- HTML MAP GENERATION ----
    print("Generating Folium Map...")
    map_center = [25.0, -89.0] # Central Gulf
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')
    
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: rgba(0,0,0,0.8); padding: 12px 24px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 16px; font-weight: bold;
         border: 1px solid rgba(255,255,255,0.2);">
        🌐 Step 6: Contiguous Traffic Networks (Per Vessel Type)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create FeatureGroups (Layers) per Ship Type
    unique_types = set(r['ship_type'] for r in routes)
    layers = {}
    for stype in unique_types:
        fg = folium.FeatureGroup(name=f"{stype} Routes", show=True)
        layers[stype] = fg
        m.add_child(fg)
        
    # Color mapping for ship types
    type_colors = {
        'Cargo': '#E74C3C',       # Red
        'Tanker': '#3498DB',      # Blue
        'Fishing': '#2ECC71',     # Green
        'Tug_Special': '#9B59B6', # Purple
        'Passenger': '#F1C40F',   # Yellow
        'WIG': '#1ABC9C',         # Teal
        'Unknown': '#95A5A6'      # Grey
    }
    
    # Draw top routes to prevent browser lagging (max 500 lines)
    routes.sort(key=lambda x: x['avg_voyages'], reverse=True)
    draw_routes = routes[:500]
    
    for route in draw_routes:
        stype = route['ship_type']
        # Handle Tug/Special underscore replacement
        color_key = stype.replace('/', '_')
        color = type_colors.get(color_key, '#95A5A6')
        
        # Extract coordinates [lat, lon]
        coords = [[pt['lat'], pt['lon']] for pt in route['points']]
        
        # Scale weight by volume (min 2, max 10)
        weight = max(2, min(10, route['avg_voyages'] / 3))
        
        popup_text = (f"<b>Route:</b> {route['itinerary']}<br>"
                     f"<b>Vessel Type:</b> {stype}<br>"
                     f"<b>Hulls Traversed:</b> {route['num_hulls']}<br>"
                     f"<b>Avg Voyages:</b> {route['avg_voyages']}")
                     
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=0.8,
            tooltip=f"{route['itinerary']} ({stype})",
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(layers[stype])
        
        # Add Port Markers
        for pt in route['points']:
            if pt['is_port']:
                folium.CircleMarker(
                    location=[pt['lat'], pt['lon']],
                    radius=5,
                    color="#FFFFFF",
                    fill=True,
                    fillColor="#FFFFFF",
                    weight=1,
                    tooltip=str(pt['id'])
                ).add_to(layers[stype])
                
    # Add layer control to toggle vessel types
    folium.LayerControl(position='topright').add_to(m)
    
    # Stats Box
    legend_items = "".join(f'<span style="color:{type_colors.get(k.replace("/", "_"), "#95A5A6")}">●</span> {k}<br>' for k in unique_types)
    stats_html = f'''
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
         background: rgba(0,0,0,0.85); padding: 15px; border-radius: 8px;
         font-family: 'Segoe UI', sans-serif; color: white; font-size: 13px;
         border: 1px solid rgba(255,255,255,0.2); max-width: 250px;">
        <b>📊 Network Stats</b><br>
        Total Rendered Routes: <b>{len(draw_routes)}</b><br>
        Line thickness represents volume<br><br>
        <b>Vessel Types:</b><br>{legend_items}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))
    
    out_map = os.path.join(OUTPUT_DIR, "step6_network_routing_map.html")
    m.save(out_map)
    print(f"Interactive Route Map saved to: {out_map}")
    elapsed = time.time() - t0
    print(f"Step 6 completed in {elapsed:.1f}s")
    
if __name__ == "__main__":
    main()
