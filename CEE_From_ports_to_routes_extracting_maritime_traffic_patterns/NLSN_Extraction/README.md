# NLSN Extraction Pipeline

This folder contains a standalone, modular end-to-end Node-Level Shipping Network (NLSN) pipeline derived from the existing PLSN source code.

## Processing Flow

1. Load full AIS trajectory data (`MMSI`, `BASEDATETIME`, `LAT`, `LON`) with memory-safe controls.
2. Resolve best PLSN clustering credentials (`k`, `r`, `neighbor_mode`, `min_dense_points`) from PLSN tuning outputs.
3. Segment trajectories per vessel by time gaps.
4. Extract feature points with Adaptive Douglas-Peucker (DP) at each gamma candidate.
5. Cluster feature points with CLIQUE.
6. Generate NLSN boundaries, nodes, and directed edges.
7. Score each gamma trial and rank the resulting NLSN.
8. Export per-gamma artifacts and comparison dashboards.

## Structure

- `config.py`: pipeline defaults.
- `main.py`: config-driven end-to-end run.
- `run_nlsn_gamma_sweep.py`: CLI runner for gamma sweeps.
- `modules/adaptive_dp.py`: DP compression + LD metrics.
- `modules/nlsn_tuner.py`: gamma sweep orchestration.
- `modules/nlsn_generator.py`: feature/node/edge exports.
- `modules/clustering.py`: CLIQUE clustering implementation.
- `modules/boundary_extractor.py`: alpha-shape boundary extraction.
- `modules/data_loader.py`: data loading utility.
- `modules/visualizer.py`: interactive HTML map output.

## Run (CLI)

From repository root:

```bash
python "CEE_From_ports_to_routes:extracting_maritime_traffic_patterns/NLSN_Extraction/run_nlsn_gamma_sweep.py" \
  --full-data-path /home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results/data/jan2025_full_minimal.parquet \
  --output-dir /home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/plsn_jan2025_31days_results/nlsn_gamma_sweep \
  --gamma-values "0.00005,0.00008,0.0001,0.00015,0.0002,0.0003,0.0004" \
  --max-input-rows 1000000 \
  --sample-mmsi-count 3000 \
  --no-cuda
```

## Run (Config-driven)

```bash
python "CEE_From_ports_to_routes:extracting_maritime_traffic_patterns/NLSN_Extraction/main.py"
```

## Outputs

Main output directory contains:

- `nlsn_gamma_sweep_results.csv`
- `nlsn_gamma_sweep_summary.json`
- `nlsn_gamma_comparison.html`
- `nlsn_run_summary.json` (for CLI runs)

Per gamma folder (`gamma_*`) contains:

- `feature_points.csv`
- `nodes.csv`
- `edges.csv`
- `boundaries.geojson` (if extractable)
- `nlsn_map.html`
- `nlsn_trial_summary.json`
