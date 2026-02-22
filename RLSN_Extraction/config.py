import os

# Base directory for the RLSN standalone project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# AIS Source Data (Parquet)
AIS_SOURCE = r"F:\PyTorch_GPU\maritime_monitoring_preprocessing\interpolated_results\consolidated_ais_data_20200105_20200112.parquet"

# NLSN Artifacts (Waypoints and Nodes for route generation)
# We pick a stable NLSN run to derive routes from
NLSN_ARTIFACTS_DIR = r"f:\PyTorch_GPU\ISLs\CEE-International_shipping_lanes_\CEE_From_ports_to_routes_extracting_maritime_traffic_patterns\PLSN_Extraction\results\nlsn\gamma_0p001_w1_1p0_w2_1p0"

# Output directory for standalone RLSN results
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# RLSN Hyperparameters
RLSN_NUM_SLICES = 15      # Density of cross-sections per edge
RLSN_SEARCH_RADIUS = 0.05 # Spatial search radius for point projection (degrees)
RLSN_MAX_POINTS = 1_000_000 # Memory limit for AIS sampling
RLSN_SIGMA_MULT = 3       # 3-sigma for 99.7% traffic coverage

# Visualization parameters
MAP_CENTER = [15.0, 85.0]
MAP_ZOOM = 4
RLSN_ROUTE_COLOR = "#0369a1" # Deep Blue
RLSN_BOUND_COLOR = "#38bdf8" # Sky Blue
