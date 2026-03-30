# GLASSEAS Pipeline — Gulf of Mexico Adaptation

This repository contains a Python adaptation of the **GLASSEAS** distributed maritime traffic pattern extraction framework. Originally developed in Scala for Apache Spark, this version is entirely ported to Python to run locally on a single machine without the heavy resource overhead of Spark, specifically optimized for processing large Parquet datasets like the Gulf of Mexico AIS data.

## Features & Pipeline Steps

The pipeline consists of 5 main steps that replicate the original Scala algorithms (DBSCAN, Haversine metrics, Lagrange interpolation, Convex Hulls) and generate interactive HTML visualizations at every stage.

### 0. Data Conversion (`convert_data.py`)
Converts raw Gulf of Mexico AIS Parquet data into the GLASSEAS CSV format. 
- Filters out pre-interpolated rows to process original raw AIS signals.
- Maps numeric vessel type codes (e.g., 70-79) to readable names (e.g., "Cargo").

### 1. Port & Waypoint Discovery (`step1_waypoints.py`)
Identifies stationary regions where ships anchor or dock.
- Filters vessels with Speed Over Ground (SOG) = 0.
- Compresses points using a high-level spatial grid.
- Applies **DBSCAN** clustering (eps=2km, minPts=10) with Haversine distance.
- Computes convex hulls around the clusters to define waypoint polygons.
- **Output:** `waypoints.csv` and `step1_waypoints_map.html`.

### 2. Voyage Segmentation (`step2_voyages.py`)
Segments continuous AIS trajectories into discrete port-to-port voyages.
- A new voyage triggers when a vessel enters a different port/waypoint.
- Large time gaps (>= 24 hours) also trigger a new voyage segment.
- Validates voyages to ensure they have an origin and destination port (`X_to_Y`).
- **Output:** `voyages.csv` and `step2_voyages_map.html`.

### 3. Trajectory Interpolation (`step3_interpolation.py`)
Fills gaps in AIS transmissions using **Lagrange Interpolation**.
- Uses a sliding window of 3 positions.
- Interpolates missing coordinates every 180 seconds (3 minutes) up to a 1-hour gap.
- Calculates synthetic bearing and speed for the interpolated points.
- **Output:** `voyages_interpolated.csv` and `step3_interpolation_map.html`.

### 4. Traffic Pattern Extraction (`step4_convex_hulls.py`)
Reconstructs common maritime routes and traffic corridors.
- Groups interpolated trajectory points by `(itinerary, grid_cell)`.
- Applies DBSCAN clustering with trajectory-aware similarity metrics (accounting for spatial distance, heading, and speed).
- Generates localized convex hulls representing dense traffic lanes.
- Extacts statistical metrics per cluster (mean heading, speed variance).
- **Output:** `convex_hulls.csv` and `step4_convex_hulls_map.html`.

### 5. Evaluation & Coverage (`step5_evaluation.py`)
Evaluates the accuracy and coverage of the extracted traffic patterns using 10-fold cross-validation.
- Tests if future/hidden trajectory points fall inside the predicted traffic pattern hulls.
- Generates a spatial density heatmap indicating high traffic vs. high speed deviation cells.
- **Output:** `evaluation_report.txt` and `step5_evaluation_map.html`.

## Requirements

```bash
pip install pandas pyarrow folium scikit-learn numpy scipy
```

## Running the Pipeline

Execute the steps sequentially from the `pipeline` directory:

```bash
python pipeline/convert_data.py
python pipeline/step1_waypoints.py
python pipeline/step2_voyages.py
python pipeline/step3_interpolation.py
python pipeline/step4_convex_hulls.py
python pipeline/step5_evaluation.py
```

All data artifacts, CSVs, and visualization HTML maps will be saved sequentially into the `pipeline/output` directory.
