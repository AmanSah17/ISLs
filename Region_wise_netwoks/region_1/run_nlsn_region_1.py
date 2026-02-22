import logging
import sys
import os

# --- Path Configuration ---
REGION_1_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add NLSN module path specifically
NLSN_MODULE_DIR = os.path.join(REPO_ROOT, "NLSN_Extraction")
if NLSN_MODULE_DIR not in sys.path:
    sys.path.insert(0, NLSN_MODULE_DIR)

# --- Imports ---
try:
    import pandas as pd
    import numpy as np
    import config_region_1 as config
    from PLSN_Extraction.modules.data_loader import AISDataLoader
    from PLSN_Extraction.modules.preprocessor import AISPreprocessor
    from NLSN_Extraction.modules.nlsn_generator import NLSNGenerator
    from NLSN_Extraction.modules.visualizer import PLSNVisualizer as NLSNVisualizer
    from NLSN_Extraction.modules.adaptive_dp import AdaptiveDPOptimizer as AdaptiveDP
    from NLSN_Extraction.modules.clustering import CLIQUEClusterer
except ImportError as e:
    print(f"CRITICIAL IMPORT ERROR: {e}")
    sys.exit(1)

def pick_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None

def main():
    if not os.path.exists(config.NLSN_OUTPUT_DIR):
        os.makedirs(config.NLSN_OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Region1_NLSN")
    
    logger.info("Starting NLSN Extraction for Region 1...")
    
    # 1. Load data
    loader = AISDataLoader(config.DATA_FILE_PATH)
    df = loader.load_data()
    
    # 2. Extract Feature Points (Waypoints) using Adaptive DP
    logger.info("Extracting feature points via Adaptive DP...")
    optimizer = AdaptiveDP()
    
    feature_points_list = []
    # Identify unique MMSIs
    mmsis = df['MMSI'].unique()
    time_col = pick_column(df, ['BaseDateTime', 'BASEDATETIME', 'Timestamp', 'DATETIME'])
    
    for mmsi in mmsis:
        vessel_df = df[df['MMSI'] == mmsi].sort_values(time_col)
        if len(vessel_df) < 5: continue # Skip short tracks
        
        points = vessel_df[['LON', 'LAT']].to_numpy()
        # Use a reasonable gamma or the optimizer
        result = optimizer.compress_with_gamma(points, gamma=0.001)
        
        # Keep only the feature points
        kept_df = vessel_df.iloc[result.keep_indices].copy()
        feature_points_list.append(kept_df)
    
    if not feature_points_list:
        logger.error("No feature points extracted.")
        return
        
    feature_points_df = pd.concat(feature_points_list)
    logger.info(f"Extracted {len(feature_points_df)} feature points from {len(df)} total points.")
    
    # 3. Clustering of Feature Points
    logger.info("Clustering feature points via CLIQUE...")
    clusterer = CLIQUEClusterer(k=30, density_threshold_r=0.00008)
    clustered_features = clusterer.fit_predict(feature_points_df)
    
    # 4. Generate NLSN Artifacts
    logger.info("Initializing NLSN Generator...")
    generator = NLSNGenerator(output_dir=config.NLSN_OUTPUT_DIR)
    generator.export_feature_points(clustered_features)
    nodes_df = generator.export_nodes_and_boundaries(None, clustered_features)
    valid_ids = set(nodes_df["port_id"].tolist())
    edges_df = generator.export_edges(clustered_features, valid_port_ids=valid_ids)
    
    # 5. Visualization (NLSN/PLSN visualizer uses generate_plsn_dashboard)
    logger.info("Generating NLSN Visualization...")
    output_map_path = os.path.join(config.NLSN_OUTPUT_DIR, "nlsn_map.html")
    visualizer = NLSNVisualizer(output_file=output_map_path)
    
    # generate_plsn_dashboard(full_df, clustered_df, boundaries_list, nodes_df, edges_df)
    visualizer.generate_plsn_dashboard(df, clustered_features, [], nodes_df, edges_df)
    
    logger.info("Region 1 NLSN Extraction completed.")

if __name__ == "__main__":
    main()
