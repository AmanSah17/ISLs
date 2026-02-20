import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Input Data ────────────────────────────────────────────────────────────────
DATA_FILE_PATH = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"

# ── Organised Output Directories ─────────────────────────────────────────────
# All pipeline artefacts are stored under results/ inside PLSN_Extraction.
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TUNING_DIR  = os.path.join(RESULTS_DIR, "tuning")
MAPS_DIR    = os.path.join(RESULTS_DIR, "maps")
RUNS_DIR    = os.path.join(RESULTS_DIR, "runs")

# Legacy single-run output paths (used by main.py)
OUTPUT_DIR         = os.path.join(RESULTS_DIR, "latest_run")
LOG_FILE           = os.path.join(OUTPUT_DIR, "pipeline.log")
NODES_CSV          = os.path.join(OUTPUT_DIR, "nodes.csv")
BOUNDARIES_GEOJSON = os.path.join(OUTPUT_DIR, "boundaries.geojson")
EDGES_CSV          = os.path.join(OUTPUT_DIR, "edges.csv")
MAP_HTML           = os.path.join(OUTPUT_DIR, "plsn_map.html")
MAP_COMPREHENSIVE_HTML = os.path.join(OUTPUT_DIR, "plsn_map_comprehensive.html")

# ── Preprocessing Parameters ─────────────────────────────────────────────────
SOG_THRESHOLD   = 0.5        # knots; vessels slower than this are considered stationary
NAV_STATUS_FILTER = [1, 5]   # 1=At Anchor, 5=Moored

# ── CLIQUE Default Parameters (paper §4.1 K=30, density ≈ r=0.00008) ────────
CLIQUE_GRID_DIVISIONS_K     = 30
CLIQUE_DENSITY_THRESHOLD_R  = 0.00008
CLIQUE_MIN_DENSE_POINTS     = None     # set int to override r-based threshold
CLIQUE_NEIGHBOR_MODE        = "4"      # "4"=Von Neumann, "8"=Moore

# ── Extensive Hyperparameter Search Space ─────────────────────────────────────
# K: from very coarse (20) to fine (1400) — includes paper's K=30
TUNING_K_VALUES = [20, 30, 50, 80, 120, 200, 300, 500, 800, 1000, 1400]

# r: density ratio from very sparse to very tight
TUNING_R_VALUES = [
    0.000005, 0.00001, 0.00002, 0.00003, 0.00005,
    0.00008,  0.0001,  0.00015, 0.0002,  0.0003,
    0.0005,   0.001
]

# Both neighbourhood modes
TUNING_NEIGHBOR_MODES = ["4", "8"]

TUNING_MIN_DENSE_POINTS = [None]     # extend with explicit counts if needed

# Scoring guide: target port count range consistent with known global data
TUNING_MIN_PORT_POINTS    = 100      # min stationary pts for a cluster to count as port
TUNING_EXPECTED_PORTS_MIN = 250
TUNING_EXPECTED_PORTS_MAX = 500
TUNING_TOP_MAPS           = 15       # number of top-scoring configs to visualise

# ── Boundary Extraction ───────────────────────────────────────────────────────
ALPHA_SHAPE_PARAMETER = 0.01

# ── Visualisation Parameters ─────────────────────────────────────────────────
VISUALIZATION_SAMPLE_SIZE = 500_000  # raw AIS heat-map points per map

# Edge / boundary rendering (applied in the enhanced visualizer)
EDGE_MIN_WIDTH    = 3    # pixels — minimum line width for any edge
EDGE_MAX_WIDTH    = 14   # pixels — maximum line width (highest-traffic edge)
BOUNDARY_WEIGHT   = 4    # pixels — port boundary polygon stroke width
NODE_MIN_RADIUS   = 5    # pixels — smallest port marker circle
NODE_MAX_RADIUS   = 18   # pixels — largest port marker circle

# Heat-map Leaflet parameters
HEATMAP_RADIUS = 10
HEATMAP_BLUR   = 15

# ── NLSN Output Directory ─────────────────────────────────────────────────────
NLSN_DIR = os.path.join(RESULTS_DIR, "nlsn")

# ── NLSN Adaptive DP Hyperparameter Sweep ─────────────────────────────────────
# gamma: Douglas-Peucker tolerance (degrees). Lower = more points kept.
# Paper recommends sweeping and picking based on LD score trade-off.
NLSN_GAMMA_VALUES = [
    0.00001, 0.00005, 0.0001, 0.0002,
    0.0005,  0.001,   0.002,  0.005,
    0.01,    0.02,    0.05,   0.1,
]

# Weight sweep for LD = w1*Dr + w2*Dl
# Dr = compression rate, Dl = distance fidelity
NLSN_W1_VALUES = [0.5, 1.0, 1.5, 2.0]   # compression weight
NLSN_W2_VALUES = [0.5, 1.0, 1.5, 2.0]   # fidelity weight

# Trajectory quality filters
NLSN_MIN_TRAJ_POINTS   = 5      # discard trajectories shorter than this
NLSN_MAX_TIME_GAP_MIN  = 720.0  # 12 h — split trajectory on longer gaps
NLSN_ALPHA_SHAPE       = 0.01   # boundary extraction alpha

# How many top-gamma configs to generate full maps for
NLSN_TOP_MAPS = 10

# Expected node count range (NLSN nodes = traffic waypoint clusters)
NLSN_EXPECTED_NODES_MIN = None
NLSN_EXPECTED_NODES_MAX = None

