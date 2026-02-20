import logging
import sys
import os
import argparse

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PARENT_DIR = os.path.dirname(BASE_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# --- Imports ---
try:
    from PLSN_Extraction import config
    from PLSN_Extraction.modules.data_loader import AISDataLoader
    from PLSN_Extraction.modules.preprocessor import AISPreprocessor
    from PLSN_Extraction.modules.clustering import CLIQUEClusterer
    from PLSN_Extraction.modules.boundary_extractor import BoundaryExtractor
    from PLSN_Extraction.modules.network_generator import PLSNGenerator
    from PLSN_Extraction.modules.visualizer import PLSNVisualizer
except ImportError as e:
    print(f"CRITICIAL IMPORT ERROR: {e}")
    sys.exit(1)

def setup_logging():
    if not os.path.exists(config.OUTPUT_DIR):
        try:
            os.makedirs(config.OUTPUT_DIR)
        except OSError as e:
            print(f"Failed to create output directory {config.OUTPUT_DIR}: {e}")
            sys.exit(1)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("Main")
    
    logger.info("Starting PLSN Extraction Pipeline (Paper-aligned CLIQUE + PLSN dashboard)...")
    logger.info(f"Running from: {os.getcwd()}")
    
    try:
        # 1. Load Data
        logger.info("Initializing Data Loader...")
        loader = AISDataLoader(config.DATA_FILE_PATH)
        df = loader.load_data()
        
        if df is None or df.empty:
            logger.error("Data loading failed or returned empty dataframe.")
            return

        # 2. Preprocess
        logger.info("Initializing Preprocessor...")
        preprocessor = AISPreprocessor(
            sog_threshold=config.SOG_THRESHOLD,
            nav_status_filter=config.NAV_STATUS_FILTER
        )
        stationary_df = preprocessor.filter_anchor_mooring(df)
        
        if stationary_df.empty:
            logger.warning("Preprocessing resulted in empty dataframe. Exiting.")
            return

        # 3. Clustering
        logger.info("Initializing Clustering...")
        clusterer = CLIQUEClusterer(
            k=config.CLIQUE_GRID_DIVISIONS_K,
            density_threshold_r=config.CLIQUE_DENSITY_THRESHOLD_R,
            min_dense_points=config.CLIQUE_MIN_DENSE_POINTS,
            neighbor_mode=config.CLIQUE_NEIGHBOR_MODE,
        )
        clustered_df = clusterer.fit_predict(stationary_df)
        
        unique_clusters = clustered_df['cluster_id'].unique()
        valid_clusters = [c for c in unique_clusters if c != -1]
        
        if not valid_clusters:
            logger.warning("No valid clusters found using current parameters.")
            return

        # 4. Boundary Extraction
        logger.info("Initializing Boundary Extractor...")
        extractor = BoundaryExtractor(alpha=config.ALPHA_SHAPE_PARAMETER)
        boundaries_list = extractor.extract_boundaries(clustered_df)
        
        if not boundaries_list:
            logger.warning("No boundaries could be extracted.")

        # 5. Network Generation
        logger.info("Initializing Network Generator...")
        generator = PLSNGenerator(output_dir=config.OUTPUT_DIR)
        nodes_df = generator.export_nodes_and_boundaries(boundaries_list, clustered_df=clustered_df)
        valid_port_ids = set(nodes_df["port_id"].astype(int).tolist()) if not nodes_df.empty else set()
        edges_df = generator.export_edges(clustered_df, valid_port_ids=valid_port_ids)
        
        # 6. Visualization (Comprehensive)
        logger.info("Initializing PLSN Dashboard Visualization...")
        visualizer = PLSNVisualizer(
            output_file=config.MAP_COMPREHENSIVE_HTML,
            sample_size=config.VISUALIZATION_SAMPLE_SIZE
        )
        visualizer.generate_plsn_dashboard(df, clustered_df, boundaries_list, nodes_df, edges_df)
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
