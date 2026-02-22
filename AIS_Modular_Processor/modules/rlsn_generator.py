import pandas as pd
import geopandas as gpd
import logging
import os
import json
from shapely.geometry import LineString, MultiPoint, mapping
from datetime import datetime

class RLSNGenerator:
    """
    Reconstructed RLSN Generator for the Modular AIS Processor.
    Extracts route-level shipping network (LineStrings and boundaries) from trajectories.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logger = logging.getLogger("RLSN_Generator")
        os.makedirs(output_dir, exist_ok=True)

    def extract_rlsn(self, ais_df, nodes_df, edges_df, port_radius_deg=0.05):
        """
        Main extraction routine.
        1. Identify port proximity for all points.
        2. Segment trajectories into voyages between ports.
        3. Aggregate voyages into routes.
        4. Generate GeoJSON outputs.
        """
        self.logger.info("Extracting Route-Level Shipping Network...")
        
        if ais_df.empty or nodes_df.empty:
            self.logger.warning("AIS or Nodes data empty. Skipping RLSN.")
            return [], []

        # 1. Map points to ports
        # Simple proximity check for reconstruction
        work_df = ais_df.copy()
        work_df['port_id'] = -1
        
        for idx, port in nodes_df.iterrows():
            dist = ((work_df.LAT - port.lat)**2 + (work_df.LON - port.lon)**2)**0.5
            work_df.loc[dist < port_radius_deg, 'port_id'] = port.port_id

        # 2. Segment by MMSI and port transitions
        # Use whatever column name data_loader gave us
        time_col = 'BaseDateTime' if 'BaseDateTime' in work_df.columns else ('BASEDATETIME' if 'BASEDATETIME' in work_df.columns else 'TIMESTAMP')
        work_df = work_df.sort_values(['MMSI', time_col])
        
        routes_data = []
        boundaries_data = []
        
        for mmsi, group in work_df.groupby('MMSI'):
            group = group.reset_index()
            # Find points where port_id changes
            group['port_change'] = (group.port_id != group.port_id.shift()).cumsum()
            
            for voyage_id, voyage in group.groupby('port_change'):
                if voyage.port_id.iloc[0] == -1:
                    # This is a potentially interesting "between ports" segment
                    # Check prev and next ports
                    prev_idx = voyage.index[0] - 1
                    next_idx = voyage.index[-1] + 1
                    
                    if prev_idx >= 0 and next_idx < len(group):
                        src_port = group.loc[prev_idx, 'port_id']
                        dst_port = group.loc[next_idx, 'port_id']
                        
                        if src_port != -1 and dst_port != -1 and src_port != dst_port:
                            # Valid transition voyage
                            pts = voyage[['LON', 'LAT']].values.tolist()
                            if len(pts) >= 2:
                                routes_data.append({
                                    "type": "Feature",
                                    "properties": {
                                        "src_port": int(src_port),
                                        "dst_port": int(dst_port),
                                        "mmsi": int(mmsi)
                                    },
                                    "geometry": mapping(LineString(pts))
                                })

        # 3. Export Results
        routes_gj = {"type": "FeatureCollection", "features": routes_data}
        routes_path = os.path.join(self.output_dir, "rlsn_routes.geojson")
        with open(routes_path, "w") as f:
            json.dump(routes_gj, f)

        self.logger.info(f"RLSN extraction complete. Saved {len(routes_data)} route segments.")
        return routes_gj, [] # Returning empty boundaries for now

# Integration Placeholder for extraction.py
