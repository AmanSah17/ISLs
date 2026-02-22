import logging
import argparse
import os
import sys
from modules.data_loader import AISDataLoader
from modules.extraction import ExtractionOrchestrator
from modules.visualizer import ModularVisualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger("AIS_Processor_CLI")

    parser = argparse.ArgumentParser(description="Modular AIS Traffic Pattern Processor")
    parser.add_argument("--ais", required=True, help="Path to AIS Parquet or CSV file")
    parser.add_argument("--region", required=True, help="Path to GeoJSON region file")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for results")
    parser.add_argument("--name", default="Analysis_Run", help="Name for this run")
    
    args = parser.parse_args()

    # 1. Load and Clip Data
    loader = AISDataLoader()
    try:
        ais_df = loader.load_ais(args.ais)
        clipped_df = loader.clip_by_region(ais_df, args.region)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    if clipped_df.empty:
        logger.warning("No AIS points found within the specified region. Exiting.")
        sys.exit(0)

    # 2. Run Extraction Pipeline
    orchestrator = ExtractionOrchestrator(args.output_dir)
    # Default config for now
    config = {
        "plsn": {"k": 1400, "r": 0.0001},
        "nlsn": {"gamma": 0.05}
    }
    results = orchestrator.run_full_pipeline(clipped_df, config)

    # 3. Visualize Results
    viz_file = os.path.join(args.output_dir, f"{args.name}_dashboard.html")
    visualizer = ModularVisualizer(viz_file)
    visualizer.set_heatmap_data(clipped_df)
    
    # Add results to visualizer (wrapped placeholders for now)
    visualizer.add_layer("macro", "PLSN Results", results['plsn'], batch="Macro")
    visualizer.add_layer("meso", "NLSN Results", results['nlsn'], batch="Meso")
    visualizer.add_layer("route", "RLSN Results", results['rlsn'], batch="Route")
    
    visualizer.generate_dashboard()
    
    logger.info(f"Analysis complete. Dashboard: {viz_file}")

if __name__ == "__main__":
    main()
