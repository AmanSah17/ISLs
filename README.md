# ISLs — CEE: International Shipping Lanes

> **From Ports to Routes: Extracting Maritime Traffic Patterns from AIS Data**
>
> A full research pipeline (Paper: *"From Ports to Routes: Extracting Maritime Traffic Patterns"*) that constructs two complementary graph representations of global maritime traffic:
> - **PLSN** — Port-Level Shipping Network (where ships stop)
> - **NLSN** — Navigation-Level Shipping Network (how ships travel between stops)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Mathematical Understanding of Algorithms](#3-mathematical-understanding-of-algorithms)
   - [3.1 AIS Data ETL & Validation](#31-ais-data-etl--validation)
   - [3.2 AIS Preprocessing — Anchoring & Mooring Filter](#32-ais-preprocessing--anchoring--mooring-filter)
   - [3.3 CLIQUE Clustering Algorithm](#33-clique-clustering-algorithm)
   - [3.4 Alpha-Shape Boundary Extraction](#34-alpha-shape-boundary-extraction)
   - [3.5 Port/Node Network Generation (PLSN / NLSN)](#35-portnode-network-generation-plsn--nlsn)
   - [3.6 Douglas-Peucker Adaptive Compression (NLSN Feature Extraction)](#36-douglas-peucker-adaptive-compression-nlsn-feature-extraction)
   - [3.7 CLIQUE Hyperparameter Tuning (PLSN)](#37-clique-hyperparameter-tuning-plsn)
   - [3.8 NLSN Gamma Sweep & Scoring](#38-nlsn-gamma-sweep--scoring)
4. [Implementation Methodology](#4-implementation-methodology)
   - [4.1 Step 0 — ETL: Build Parquet](#41-step-0--etl-build-parquet)
   - [4.2 Step 1 — PLSN Pipeline](#42-step-1--plsn-pipeline)
   - [4.3 Step 2 — NLSN Pipeline](#43-step-2--nlsn-pipeline)
5. [Dependencies & Installation](#5-dependencies--installation)
6. [Quick-Start Usage](#6-quick-start-usage)
7. [Output Artifacts](#7-output-artifacts)
8. [Key Configuration Parameters](#8-key-configuration-parameters)

---

## 1. Project Overview

Maritime Automatic Identification System (AIS) broadcasts vessel positions at regular intervals. This project transforms raw AIS CSV files into a **dual-layer shipping network graph**:

| Network | What it represents | Core Algorithm |
|---|---|---|
| **PLSN** | Ports as nodes; vessel port-call transitions as edges | CLIQUE density clustering |
| **NLSN** | Navigation waypoints as nodes; route segments as edges | Adaptive Douglas-Peucker + CLIQUE |

The pipeline is structured as a cascade:

```
Raw AIS CSVs ──► ETL Parquet ──► PLSN (ports discovered) ──► NLSN (routes discovered)
```

---

## 2. Folder Structure

```
ISLs/
├── build_plsn_nlsn_parquet.py              # Step 0: Memory-safe AIS ETL → Parquet
│
└── CEE-International_shipping_lanes_/
    ├── run_ais_pipeline_31days.py           # Orchestrates the full 31-day pipeline
    ├── run_january31_plsn.py               # PLSN-only run script for January
    ├── run_january31_nlsn.py               # NLSN-only run script for January
    │
    └── CEE_From_ports_to_routes_extracting_maritime_traffic_patterns/
        ├── main.pdf                         # Original research paper (PDF)
        │
        ├── PLSN_Extraction/                 # Port-Level Shipping Network
        │   ├── __init__.py
        │   ├── config.py                    # All PLSN parameters & paths
        │   ├── main.py                      # PLSN pipeline entry point
        │   ├── run_hyperparam_comparison.py # Hyperparameter comparison runner
        │   ├── run_nlsn_gamma_sweep.py      # NLSN gamma sweep (from PLSN context)
        │   ├── tune_clique.py               # Standalone CLIQUE tuner
        │   └── modules/
        │       ├── __init__.py
        │       ├── data_loader.py           # Parquet/CSV data ingestion
        │       ├── preprocessor.py          # AIS anchor/mooring filter (Eq.1, Eq.2)
        │       ├── clustering.py            # CLIQUE clusterer (GPU/CPU)
        │       ├── boundary_extractor.py    # Alpha-shape polygon extraction
        │       ├── network_generator.py     # PLSNGenerator: nodes.csv, edges.csv
        │       ├── hyperparameter_tuner.py  # Systematic CLIQUE grid search
        │       ├── nlsn_generator.py        # NLSNGenerator (shared module)
        │       ├── adaptive_dp.py           # Douglas-Peucker compression
        │       ├── nlsn_tuner.py            # NLSNGammaTuner: gamma sweep
        │       ├── visualizer.py            # Folium/HTML dashboard
        │       └── 01_CLIQUE_algorithms.ipynb  # Interactive exploration notebook
        │
        └── NLSN_Extraction/                 # Navigation-Level Shipping Network
            ├── __init__.py
            ├── config.py                    # All NLSN parameters & paths
            ├── main.py                      # NLSN pipeline entry point
            ├── run_nlsn_gamma_sweep.py      # Standalone gamma sweep runner
            └── modules/
                ├── __init__.py
                ├── data_loader.py           # Data ingestion (shared)
                ├── preprocessor.py          # AIS filter (shared)
                ├── clustering.py            # CLIQUE clusterer (shared)
                ├── boundary_extractor.py    # Alpha-shape (shared)
                ├── nlsn_generator.py        # NLSNGenerator: nodes/edges export
                ├── adaptive_dp.py           # Douglas-Peucker (shared)
                ├── nlsn_tuner.py            # NLSNGammaTuner sweep
                └── visualizer.py            # Dashboard visualizer (shared)
```

---

## 3. Mathematical Understanding of Algorithms

### 3.1 AIS Data ETL & Validation

**File:** `build_plsn_nlsn_parquet.py`

The ETL pipeline ingests raw daily AIS CSVs and produces a clean Parquet file. Each row is validated using **vectorized domain filters** (Numba JIT for CPU, CUDA kernels for GPU):

#### Validity Constraints

For each AIS record with fields *(lat, lon, sog, cog, nav)*:

$$\text{valid} = \underbrace{-90 \le \text{lat} \le 90}_{\text{latitude}} \;\wedge\; \underbrace{-180 \le \text{lon} \le 180}_{\text{longitude}} \;\wedge\; \underbrace{0 \le \text{sog} \le 102.4}_{\text{speed (knots)}} \;\wedge\; \underbrace{0 \le \text{cog} \le 360}_{\text{course}} \;\wedge\; \underbrace{0 \le \text{nav} \le 15}_{\text{nav status}}$$

The upper bound 102.4 knots corresponds to the AIS "not available" sentinel value.

**CUDA Kernel:** Each thread `i` independently evaluates the combined validity predicate, writing `out[i] ∈ {0,1}`. The CPU-parallel version uses Numba `@njit(parallel=True)` with `prange`.

**Output Schema:**

| Column | Type | Description |
|---|---|---|
| MMSI | int64 | Maritime Mobile Service Identity |
| BASEDATETIME | timestamp[ns] | UTC timestamp |
| LAT | float32 | WGS-84 latitude |
| LON | float32 | WGS-84 longitude |
| SOG | float32 | Speed over ground (knots) |
| COG | float32 | Course over ground (degrees) |
| NAVSTATUS | int16 | AIS navigational status code |

---

### 3.2 AIS Preprocessing — Anchoring & Mooring Filter

**File:** `modules/preprocessor.py`  
**Paper equations:** Eq. (1) and Eq. (2)

Only stationary vessel observations are relevant for port detection. The filter extracts:

$$P_{\text{anchoring}} = \{ p \mid \text{NAVSTATUS}(p) = 1 \;\wedge\; \text{SOG}(p) < \tau \}$$

$$P_{\text{mooring}} = \{ p \mid \text{NAVSTATUS}(p) = 5 \;\wedge\; \text{SOG}(p) < \tau \}$$

where **τ = 0.5 knots** (default threshold).

The AIS navigational status codes:
- **1** = At anchor  
- **5** = Moored  

The output is the union: $P_{\text{stationary}} = P_{\text{anchoring}} \cup P_{\text{mooring}}$

An additional **quality mask** is applied requiring valid MMSI, valid lat/lon bounds, and non-null timestamp.

---

### 3.3 CLIQUE Clustering Algorithm

**File:** `modules/clustering.py`

CLIQUE (CLustering In QUEst) is a grid-based density clustering algorithm for spatial subspaces. The implementation follows the paper parameters:

#### Step 1 — Equal-Width Partitioning

The 2D geographic space [LON, LAT] is divided into a **K × K** grid:

$$x_i = \text{clip}\!\left(\left\lfloor \frac{\text{lon}_i - \text{lon}_{\min}}{\text{lon}_{\text{span}}} \cdot K \right\rfloor, 0, K-1\right)$$

$$y_i = \text{clip}\!\left(\left\lfloor \frac{\text{lat}_i - \text{lat}_{\min}}{\text{lat}_{\text{span}}} \cdot K \right\rfloor, 0, K-1\right)$$

**GPU acceleration:** A CUDA kernel assigns grid indices $(x_i, y_i)$ in parallel across all points.

#### Step 2 — Density Threshold

A grid unit $(x, y)$ is called **dense** if the number of points it contains exceeds the threshold:

$$\text{dense\_threshold} = \begin{cases} \text{min\_dense\_points} & \text{if explicitly set} \\ \lceil r \cdot N \rceil & \text{otherwise} \end{cases}$$

where $r$ is the density ratio parameter and $N$ is the total number of stationary points.

#### Step 3 — Connected Components (BFS)

Dense units are connected via BFS/DFS using either:
- **4-connectivity** (Von Neumann neighbourhood): $(x±1, y)$ and $(x, y±1)$
- **8-connectivity** (Moore neighbourhood): all 8 adjacent cells

Each connected component of dense units forms one **cluster** (port region). Points in non-dense units receive `cluster_id = -1` (noise).

**Complexity:** $O(N)$ for grid assignment + $O(K^2)$ for BFS over dense units.

---

### 3.4 Alpha-Shape Boundary Extraction

**File:** `modules/boundary_extractor.py`

For each port cluster, a **polygon boundary** is extracted using the **Alpha-Shape** algorithm (a generalization of the convex hull):

The alpha-shape with parameter α is the boundary of the union of all closed disks of radius $r = 1/\alpha$ that contain the point set. Formally:

$$\alpha\text{-shape}(P, \alpha) = \text{boundary}\!\left(\bigcap_{r=1/\alpha} \{B(c,r) : B(c,r) \supseteq P\}^c\right)$$

- **Lower α** → larger disks → shape closer to convex hull (smoother boundary)
- **Higher α** → smaller disks → tighter, more concave boundaries

The implementation uses the `alphashape` Python library, with fallback to `Shapely`'s convex hull when the library is unavailable. Degenerate clusters (< 4 unique points, collinear points) are skipped.

---

### 3.5 Port/Node Network Generation (PLSN / NLSN)

**Files:** `modules/network_generator.py` (PLSN), `modules/nlsn_generator.py` (NLSN)

#### Nodes
Each non-noise cluster becomes a **node** (port). The node position is the geometric centroid of its alpha-shape polygon:

$$\text{centroid}(P) = \left(\frac{1}{|P|}\sum_{p \in P} \text{lon}_p,\; \frac{1}{|P|}\sum_{p \in P} \text{lat}_p\right)$$

#### Directed Edges (Port-Call Transitions)

For each vessel (MMSI), the sequence of visited port clusters (sorted by time) defines **directed transitions**. Consecutive duplicate port visits are collapsed:

$$e_{u \to v} = \{(\text{MMSI}, C_t, C_{t+1}) \mid C_t \ne C_{t+1}, \; C_t, C_{t+1} \ne -1\}$$

Edges are aggregated to compute:
- `transition_count` — total number of times the route $u \to v$ was observed
- `unique_vessels` — number of distinct vessels that used the route

---

### 3.6 Douglas-Peucker Adaptive Compression (NLSN Feature Extraction)

**File:** `modules/adaptive_dp.py`

For NLSN, full AIS trajectories are compressed to **feature points** (key waypoints) using the **Douglas-Peucker (DP)** polyline simplification algorithm.

#### Classic Douglas-Peucker

Given a polyline of $N$ points and tolerance $\gamma$:

1. Keep the first and last points.
2. Find the point $p^*$ with **maximum perpendicular distance** to the line segment connecting the current endpoints.
3. If $d(p^*, \text{segment}) > \gamma$: mark $p^*$ as a split point; recurse on both sub-polylines.
4. If $d(p^*, \text{segment}) \le \gamma$: discard all interior points.

**Perpendicular distance** from point $p$ to segment $\overrightarrow{AB}$:

$$d(p, \overrightarrow{AB}) = \frac{\|(p - A) \times (B - A)\|}{\|B - A\|}$$

implemented via the projected scalar: $t = \text{clip}\!\left(\frac{(p-A)\cdot(B-A)}{\|B-A\|^2}, 0, 1\right)$, then $d = \|p - (A + t(B-A))\|$.

#### Adaptive DP Scoring: LD Score

The quality of a given compression is measured by:

$$D_r = 1 - \frac{n}{N}$$

The **compression rate**: fraction of points removed ($n$ = compressed count, $N$ = original count).

$$D_l = 1 - \frac{|\text{len}(\text{compressed}) - \text{len}(\text{original})|}{\text{len}(\text{original})}$$

The **distance similarity**: how well the total polyline length is preserved.

$$\text{LD} = w_1 \cdot D_r + w_2 \cdot D_l$$

where $w_1 = w_2 = 1.0$ by default (equal weight to compression and fidelity).

The **gamma sweep** evaluates multiple $\gamma$ values and selects the one that maximizes the LD score:

$$\gamma^* = \arg\max_{\gamma \in \Gamma} \text{LD}(\gamma)$$

#### Trajectory Segmentation

Before compression, each vessel's track is split into **trajectories** whenever the time gap between consecutive AIS pings exceeds `max_time_gap_minutes` (default: 720 min = 12 hours). Only trajectories with ≥ `min_trajectory_points` points are retained.

---

### 3.7 CLIQUE Hyperparameter Tuning (PLSN)

**File:** `modules/hyperparameter_tuner.py`

A systematic grid search over the CLIQUE parameter space:

$$\Omega = \{K\} \times \{r\} \times \{\text{neighbor\_mode}\} \times \{\text{min\_dense\_points}\}$$

Each trial $(K, r, m, d)$ is scored by a **composite objective**:

$$\text{score} = 0.50 \cdot S_{\text{count}} + 0.25 \cdot S_{\text{coverage}} + 0.20 \cdot S_{\text{balance}} + 0.05 \cdot S_{\text{speed}}$$

where:

| Component | Formula | Meaning |
|---|---|---|
| $S_{\text{count}}$ | $\exp\!\left(-\frac{\min(\|n_p - N_{\min}\|, \|n_p - N_{\max}\|)}{\max(10, N_{\max}-N_{\min})}\right)$ | Port count alignment with expected range $[N_{\min}, N_{\max}]$ |
| $S_{\text{coverage}}$ | $\frac{\text{clustered points}}{N}$ | Fraction of stationary points assigned to a cluster |
| $S_{\text{balance}}$ | $1 - \frac{\text{largest cluster size}}{\text{total clustered}}$ | Penalises one dominant mega-cluster |
| $S_{\text{speed}}$ | $\frac{1}{1 + t_{\text{runtime}}}$ | Efficiency preference |

Results are sorted by score. The top configuration is used for subsequent NLSN extraction.

---

### 3.8 NLSN Gamma Sweep & Scoring

**File:** `modules/nlsn_tuner.py`

After trajectory preparation, each gamma value $\gamma \in \Gamma$ is evaluated end-to-end:

$$\text{score}(\gamma) = 0.35 \cdot \overline{\text{LD}} + 0.20 \cdot C_{\text{coverage}} + 0.15 \cdot S_{\text{balance}} + 0.15 \cdot S_{\text{node}} + 0.10 \cdot S_{\text{density}} + 0.05 \cdot S_{\text{transition}}$$

| Term | Formula | Meaning |
|---|---|---|
| $\overline{\text{LD}}$ | Global LD score across all trajectories | Compression-fidelity quality |
| $C_{\text{coverage}}$ | $\frac{\text{feature pts in clusters}}{\text{total feature pts}}$ | Fraction of waypoints assigned to nodes |
| $S_{\text{balance}}$ | $1 - \text{largest cluster share}$ | Balanced node sizes |
| $S_{\text{node}}$ | $\exp(-\|n_{\text{nodes}} - N_{\text{target}}\| / w)$ | Node count alignment |
| $S_{\text{density}}$ | $1 - \exp(-25 \cdot \frac{E}{N(N-1)})$ | Edge density (graph connectedness) |
| $S_{\text{transition}}$ | $\min(1,\; \frac{\bar{t}}{3})$ | Average transitions per edge |

The best $\gamma^*$ is selected and a full interactive HTML comparison report is generated.

---

## 4. Implementation Methodology

### 4.1 Step 0 — ETL: Build Parquet

```bash
python build_plsn_nlsn_parquet.py \
  --input-glob "path/to/AIS_2020_01_*.csv" \
  --output-parquet "path/to/output.parquet" \
  --chunk-size 300000 \
  --compression zstd \
  [--use-cuda]          # optional GPU acceleration
```

**Methodology:**
1. Glob-match all daily CSV files; stream them in `chunk_size` batches.
2. Standardise column names: `BaseDateTime → BASEDATETIME`, `Status → NAVSTATUS`, etc.
3. Coerce types; drop rows with nulls in any of the 7 required columns.
4. Apply Numba/CUDA validity mask; write valid rows to a single Parquet using Apache Arrow + `ParquetWriter` (streaming, memory-safe).
5. Log per-chunk drop counts by category (null, geo, nav, sog, cog).

---

### 4.2 Step 1 — PLSN Pipeline

```bash
# Option A: direct entry point
python -m PLSN_Extraction.main

# Option B: hyperparameter sweep first
python PLSN_Extraction/tune_clique.py

# Option C: 31-day batch pipeline
python run_january31_plsn.py
```

**Full pipeline (PLSN/main.py) stages:**

```
1. AISDataLoader.load_data()           → pandas DataFrame from Parquet
2. AISPreprocessor.filter_anchor_mooring() → stationary points only
3. CLIQUEClusterer.fit_predict()       → cluster_id per point
4. BoundaryExtractor.extract_boundaries() → alpha-shape polygons
5. PLSNGenerator.export_nodes_and_boundaries() → nodes.csv + boundaries.geojson
6. PLSNGenerator.export_edges()        → edges.csv (directed port transitions)
7. PLSNVisualizer.generate_plsn_dashboard() → interactive HTML map
```

The **hyperparameter tuner** (Step 3 override) runs a full grid search over CLIQUE parameters and writes:
- `clique_tuning_results.csv` — all trial scores
- `clique_tuning_summary.json` — best configuration

---

### 4.3 Step 2 — NLSN Pipeline

```bash
# Option A: standalone NLSN pipeline
python -m NLSN_Extraction.main

# Option B: gamma sweep runner
python NLSN_Extraction/run_nlsn_gamma_sweep.py

# Option C: 31-day batch pipeline
python run_january31_nlsn.py

# Option D: full combined pipeline
python run_ais_pipeline_31days.py
```

**Full pipeline (NLSN) stages:**

```
1. Load full AIS data (all moving + stationary points)
2. Load best PLSN parameters (k, r, neighbor_mode) from PLSN sweep results
3. NLSNGammaTuner.run_sweep():
   For each γ in Γ:
     a. _prepare_trajectories()          → segment tracks at time gaps ≥ 12h
     b. _extract_feature_points_for_gamma() → DP compression w/ gamma γ
     c. CLIQUEClusterer.fit_predict()    → cluster feature points → navigation nodes
     d. BoundaryExtractor               → node polygon boundaries
     e. NLSNGenerator.export_nodes_and_boundaries() → nodes.csv
     f. NLSNGenerator.export_edges()    → edges.csv (route transitions)
     g. PLSNVisualizer                  → per-gamma HTML map
     h. _score_trial()                  → composite score
4. Sort by score → best γ* identified
5. Export: nlsn_gamma_sweep_results.csv, nlsn_gamma_sweep_summary.json
6. Build: nlsn_gamma_comparison.html (interactive gamma comparison dashboard)
```

---

## 5. Dependencies & Installation

### Core Requirements

```bash
pip install pandas numpy pyarrow geopandas shapely alphashape folium tqdm
```

### GPU Acceleration (optional)

```bash
# CUDA-capable NVIDIA GPU required
pip install numba
# Install CUDA toolkit matching your GPU driver (see https://developer.nvidia.com/cuda-downloads)
```

### Full Environment

```bash
pip install pandas numpy pyarrow geopandas shapely alphashape folium tqdm numba psutil
```

### Python Version

Requires Python **≥ 3.10** (uses `int | None` union type syntax throughout).

---

## 6. Quick-Start Usage

### Minimal PLSN run (CPU, no GPU required)

1. Edit `PLSN_Extraction/config.py`:
   ```python
   DATA_FILE_PATH = "/path/to/your_ais_data.parquet"
   OUTPUT_DIR = "/path/to/output"
   ```

2. Run:
   ```bash
   cd CEE-International_shipping_lanes_/CEE_From_ports_to_routes_extracting_maritime_traffic_patterns
   python -m PLSN_Extraction.main
   ```

### Minimal NLSN run (after PLSN tuning)

1. Edit `NLSN_Extraction/config.py`:
   ```python
   FULL_DATA_PATH = "/path/to/your_ais_data.parquet"
   PLSN_SUMMARY_JSON = "/path/to/plsn_results/clique_tuning_summary.json"
   PLSN_RESULTS_CSV = "/path/to/plsn_results/clique_tuning_results.csv"
   OUTPUT_DIR = "/path/to/nlsn_output"
   ```

2. Run:
   ```bash
   python -m NLSN_Extraction.main
   ```

---

## 7. Output Artifacts

| File | Location | Description |
|---|---|---|
| `output.parquet` | ETL output path | Cleaned, standardised AIS data |
| `nodes.csv` | `OUTPUT_DIR/` | Port/node centroids with lat, lon, feature_points, area |
| `boundaries.geojson` | `OUTPUT_DIR/` | Alpha-shape polygon for each cluster/port |
| `edges.csv` | `OUTPUT_DIR/` | Directed edges: source, target, transition_count, unique_vessels |
| `plsn_map_comprehensive.html` | `OUTPUT_DIR/` | Interactive Folium dashboard (heatmap + network) |
| `clique_tuning_results.csv` | tuning dir | All CLIQUE hyperparameter trial results |
| `clique_tuning_summary.json` | tuning dir | Best CLIQUE configuration |
| `nlsn_gamma_sweep_results.csv` | nlsn output | All gamma trial results |
| `nlsn_gamma_sweep_summary.json` | nlsn output | Best gamma configuration |
| `nlsn_gamma_comparison.html` | nlsn output | Interactive gamma comparison dashboard |
| `gamma_<X>/feature_points.csv` | per-gamma dir | DP-compressed waypoints for each gamma |
| `gamma_<X>/nlsn_trial_summary.json` | per-gamma dir | Full metrics for each gamma trial |

---

## 8. Key Configuration Parameters

### CLIQUE (Shared — both PLSN & NLSN)

| Parameter | Default (PLSN) | Default (NLSN) | Description |
|---|---|---|---|
| `K` | 1400 | 1400 | Grid divisions per dimension |
| `r` | 0.00001 | 0.00008 | Density threshold ratio |
| `neighbor_mode` | `"4"` | `"4"` | `"4"` = Von Neumann, `"8"` = Moore |
| `min_dense_points` | `None` | `None` | Override r-based threshold |

### NLSN — DP Compression

| Parameter | Default | Description |
|---|---|---|
| `gamma_values` | `[5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4]` | Epsilon values to sweep |
| `min_trajectory_points` | 5 | Minimum points to keep a trajectory |
| `max_time_gap_minutes` | 720 | Split trajectory on gaps > 12 hours |
| `w1` | 1.0 | Compression rate weight in LD score |
| `w2` | 1.0 | Distance similarity weight in LD score |
| `alpha_shape` | 0.01 | Alpha for boundary extraction |

### AIS Preprocessing

| Parameter | Default | Description |
|---|---|---|
| `sog_threshold` | 0.5 knots | Maximum SOG to be classified as stationary |
| `nav_status_filter` | `[1, 5]` | AIS status codes: 1=Anchor, 5=Moored |

---

## References

- Zhao, L., et al. *"From Ports to Routes: Extracting Maritime Traffic Patterns from AIS Data."* (See `main.pdf` in repository.)
- Agrawal, R., Gehrke, J., et al. *"Automatic Subspace Clustering of High Dimensional Data."* — CLIQUE algorithm.
- Douglas, D.H., Peucker, T.K. *"Algorithms for the reduction of the number of points required to represent a digitized line or its caricature."* The Canadian Cartographer. 1973.
- Edelsbrunner, H., Kirkpatrick, D., Seidel, R. *"On the shape of a set of points in the plane."* IEEE Trans. Inf. Theory. 1983. — Alpha-shapes.

---

*Generated automatically to accompany the ISLs—CEE codebase.*
