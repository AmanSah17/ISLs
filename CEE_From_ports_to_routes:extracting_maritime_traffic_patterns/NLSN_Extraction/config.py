import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/Output defaults for January 2025 full-month workflow.
FULL_DATA_PATH = (
    "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/"
    "plsn_jan2025_31days_results/data/jan2025_full_minimal.parquet"
)
PLSN_SUMMARY_JSON = (
    "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/"
    "plsn_jan2025_31days_results/tuning/expanded_hyperparam_sweep_summary.json"
)
PLSN_RESULTS_CSV = (
    "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/"
    "plsn_jan2025_31days_results/tuning/expanded_hyperparam_sweep_results.csv"
)
OUTPUT_DIR = (
    "/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/"
    "plsn_jan2025_31days_results/nlsn_gamma_sweep"
)

# NLSN gamma search defaults.
GAMMA_VALUES = [0.00005, 0.00008, 0.0001, 0.00015, 0.0002, 0.0003, 0.0004]
MIN_TRAJECTORY_POINTS = 5
MAX_TIME_GAP_MINUTES = 720.0
WEIGHT_W1 = 1.0
WEIGHT_W2 = 1.0
ALPHA_SHAPE = 0.01
MAP_SAMPLE_SIZE = 120000

# Memory-safety defaults.
MAX_INPUT_ROWS = 5_000_000
SAMPLE_MMSI_COUNT = 0

# Runtime defaults.
REQUIRE_CUDA = False
SHOW_PROGRESS = True

# Compatibility fallback constants used by runner when PLSN summary is missing.
ALPHA_SHAPE_PARAMETER = ALPHA_SHAPE
CLIQUE_GRID_DIVISIONS_K = 1400
CLIQUE_DENSITY_THRESHOLD_R = 0.00008
CLIQUE_NEIGHBOR_MODE = "4"
CLIQUE_MIN_DENSE_POINTS = None
