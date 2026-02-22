import os

# Project root directory for Region 1
REGION_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Input Data ────────────────────────────────────────────────────────────────
# Use the filtered parquet for Region 1
DATA_FILE_PATH = os.path.join(REGION_DIR, "region_1_ais.parquet")

# ── Organised Output Directories ─────────────────────────────────────────────
PLSN_DIR = os.path.join(REGION_DIR, "PLSN_Extraction")
RESULTS_DIR = os.path.join(PLSN_DIR, "results")
TUNING_DIR  = os.path.join(RESULTS_DIR, "tuning")
MAPS_DIR    = os.path.join(RESULTS_DIR, "maps")
RUNS_DIR    = os.path.join(RESULTS_DIR, "runs")

# Legacy single-run output paths (used by main scripts)
OUTPUT_DIR         = os.path.join(RESULTS_DIR, "latest_run")

LOG_FILE           = os.path.join(OUTPUT_DIR, "pipeline.log")
NODES_CSV          = os.path.join(OUTPUT_DIR, "nodes.csv")
BOUNDARIES_GEOJSON = os.path.join(OUTPUT_DIR, "boundaries.geojson")
EDGES_CSV          = os.path.join(OUTPUT_DIR, "edges.csv")
MAP_HTML           = os.path.join(OUTPUT_DIR, "plsn_map.html")
MAP_COMPREHENSIVE_HTML = os.path.join(OUTPUT_DIR, "plsn_map_comprehensive.html")

# ── Preprocessing Parameters (Keep standard or adjust if needed) ─────────────
SOG_THRESHOLD   = 0.5
NAV_STATUS_FILTER = [1, 5]

# ── CLIQUE Parameters ────────────────────────────────────────────────────────
# Coarsening grid slightly as the area is smaller, but keeping paper defaults first
CLIQUE_GRID_DIVISIONS_K     = 1400
CLIQUE_DENSITY_THRESHOLD_R  = 1e-05
CLIQUE_MIN_DENSE_POINTS     = None
CLIQUE_NEIGHBOR_MODE        = "4"

# ── Extensive Hyperparameter Search Space ─────────────────────────────────────
TUNING_K_VALUES = [20, 30, 50, 80, 120, 200, 300, 500, 800, 1000, 1400]
TUNING_R_VALUES = [
    0.000005, 0.00001, 0.00002, 0.00003, 0.00005,
    0.00008,  0.0001,  0.00015, 0.0002,  0.0003,
    0.0005,   0.001
]
TUNING_NEIGHBOR_MODES = ["4", "8"]
TUNING_MIN_DENSE_POINTS = [None]
TUNING_MIN_PORT_POINTS    = 100
TUNING_EXPECTED_PORTS_MIN = 30 # Adjusted for smaller region
TUNING_EXPECTED_PORTS_MAX = 100 # Adjusted for smaller region
TUNING_TOP_MAPS           = 10

# ── Boundary Extraction ───────────────────────────────────────────────────────
ALPHA_SHAPE_PARAMETER = 0.01

# ── Visualisation Parameters ─────────────────────────────────────────────────
VISUALIZATION_SAMPLE_SIZE = 100_000 # Smaller sample for smaller region

EDGE_MIN_WIDTH    = 3
EDGE_MAX_WIDTH    = 14
BOUNDARY_WEIGHT   = 4
NODE_MIN_RADIUS   = 5
NODE_MAX_RADIUS   = 18
HEATMAP_RADIUS = 10
HEATMAP_BLUR   = 15

# ── NLSN / RLSN Paths ────────────────────────────────────────────────────────
RLSN_OUTPUT_DIR = os.path.join(REGION_DIR, "RLSN_Extraction", "results")
NLSN_DIR = os.path.join(REGION_DIR, "NLSN_Extraction")
NLSN_RESULTS_DIR = os.path.join(NLSN_DIR, "results")
NLSN_OUTPUT_DIR = NLSN_RESULTS_DIR # Legacy

# ── NLSN Adaptive DP Hyperparameter Sweep ─────────────────────────────────────
NLSN_GAMMA_VALUES = [
    0.00001, 0.00005, 0.0001, 0.0002,
    0.0005,  0.001,   0.002,  0.005,
    0.01,    0.02,    0.05,   0.1,
]
NLSN_W1_VALUES = [0.5, 1.0, 1.5, 2.0]
NLSN_W2_VALUES = [0.5, 1.0, 1.5, 2.0]
NLSN_MIN_TRAJ_POINTS   = 5
NLSN_MAX_TIME_GAP_MIN  = 720.0
NLSN_ALPHA_SHAPE       = 0.01
NLSN_TOP_MAPS = 5
NLSN_EXPECTED_NODES_MIN = None
NLSN_EXPECTED_NODES_MAX = None
