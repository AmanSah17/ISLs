import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Data
DATA_FILE_PATH = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/processed/ais_data_3day.parquet"

# Output Directory
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline.log")
NODES_CSV = os.path.join(OUTPUT_DIR, "nodes.csv")
BOUNDARIES_GEOJSON = os.path.join(OUTPUT_DIR, "boundaries.geojson")
EDGES_CSV = os.path.join(OUTPUT_DIR, "edges.csv")
MAP_HTML = os.path.join(OUTPUT_DIR, "plsn_map.html")
MAP_COMPREHENSIVE_HTML = os.path.join(OUTPUT_DIR, "plsn_map_comprehensive.html")

# Preprocessing Parameters
SOG_THRESHOLD = 0.5  # Knots
NAV_STATUS_FILTER = [1, 5]  # 1=At Anchor, 5=Moored

# CLIQUE Parameters (paper notation)
CLIQUE_GRID_DIVISIONS_K = 1400
CLIQUE_DENSITY_THRESHOLD_R = 0.00001
CLIQUE_MIN_DENSE_POINTS = None  # Set int to override r-based threshold.
CLIQUE_NEIGHBOR_MODE = "4"  # "4" for edge-connected dense units, "8" for Moore neighborhood.

# Broad hyperparameter search space (very small -> very large)
TUNING_K_VALUES = [60, 80, 120, 160, 240, 360, 500, 800, 1000, 1400]
TUNING_R_VALUES = [0.00001, 0.00002, 0.00003, 0.00005, 0.00008, 0.0001, 0.00015, 0.0002, 0.0003, 0.0004]
TUNING_NEIGHBOR_MODES = ["4", "8"]
TUNING_MIN_DENSE_POINTS = [None]
TUNING_MIN_PORT_POINTS = 100
TUNING_EXPECTED_PORTS_MIN = 250
TUNING_EXPECTED_PORTS_MAX = 500
TUNING_TOP_MAPS = 12

# Boundary Extraction Parameters
ALPHA_SHAPE_PARAMETER = 0.01

# Visualization Parameters
VISUALIZATION_SAMPLE_SIZE = 500000 # Increased for HeatMap coverage
HEATMAP_RADIUS = 10
HEATMAP_BLUR = 15
