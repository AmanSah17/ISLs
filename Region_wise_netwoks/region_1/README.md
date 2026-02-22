# Region 1: Maritime Traffic Pattern Analysis

This directory contains the replicated pipeline and results for **Region 1**, a custom maritime area of interest. The analysis follows a multi-scale approach, extracting Port-Level (PLSN), Meso-scale Waypoint (NLSN), and Route-Level (RLSN) shipping networks.

## 📊 Analytics Summary

The analysis was performed using an extensive hyperparameter sweep to ensure optimal pattern discovery.

- **Data Volume**: 216,164 AIS records filtered for Region 1.
- **PLSN Discovery**: 264 configurations tested. 
    - **Best Config**: trial #243 (K=1400, r=1e-05).
    - **Discovery Score**: 0.989.
- **NLSN meso-scale**: Optimal Douglas-Peucker gamma identified at **0.02** for trajectory compression.
- **RLSN extracted**: High-fidelity route corridors and customary boundaries generated.

### [Interactive Dashboard]
Access the unified research portal for Region 1:
👉 **[integrated_dashboard_region_1.html](integrated_dashboard_region_1.html)**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual Environment: `F:\PyTorch_GPU\torch_gpu\Scripts\activate.bat`
- Dependencies: `pandas`, `geopandas`, `shapely`, `numba`, `scipy`

### Project Structure
```text
region_1/
├── region_1.geojson           # ROI Boundary
├── region_1_ais.parquet       # Filtered AIS Data
├── config_region_1.py         # Tailored Configuration
├── run_plsn_tuning_region_1.py # PLSN Hyperparam Sweep
├── run_nlsn_tuning_region_1.py # NLSN Gamma Sweep
├── run_rlsn_region_1.py       # RLSN Extraction Driver
├── visualize_region_1.py      # Integrated Dashboard Generator
├── PLSN_Extraction/           # PLSN Results & Maps
├── NLSN_Extraction/           # NLSN Results & Waypoints
└── RLSN_Extraction/           # RLSN Corridors & Bounds
```

## 🛠 Execution Steps

1. **Setup Environment**:
   ```powershell
   F:\PyTorch_GPU\torch_gpu\Scripts\activate.bat
   ```

2. **Data Filtering**:
   ```powershell
   python filter_region_1_data.py
   ```

3. **Hyperparameter Tuning (PLSN)**:
   ```powershell
   python run_plsn_tuning_region_1.py
   ```

4. **Waypoint Extraction (NLSN)**:
   ```powershell
   python run_nlsn_tuning_region_1.py
   ```

5. **Route Corollary Generation (RLSN)**:
   ```powershell
   python run_rlsn_region_1.py
   ```

6. **Generate Final Dashboard**:
   ```powershell
   python visualize_region_1.py
   ```

---

Aman Sah.
