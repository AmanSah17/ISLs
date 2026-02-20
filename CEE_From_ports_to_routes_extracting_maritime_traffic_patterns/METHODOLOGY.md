# Methodology: Multiscale Maritime Network Extraction

This document details the technical approach used to extract and visualize maritime networks at two distinct scales: **Macro (PLSN)** and **Meso (NLSN)**.

---

## 1. Macro-scale: Ports to Shipping Lanes (PLSN)
The PLSN scale identifies major global ports and the deep-sea shipping lanes that connect them.

### Stationary Point Detection (Anchoring & Mooring)
We utilize **AIS Message 1 (Dynamic Report)** and **Message 5 (Static and Voyage Data)** to identify vessels that are not in transit.
*   **Anchoring (NAVSTATUS 1)**: Vessels waiting at sea near a port.
*   **Mooring (NAVSTATUS 5)**: Vessels physically at a berth or jetty.

**Classification Heuristics:**
Points are classified as "Stationary" if they satisfy:
1.  `NAVSTATUS` is either `1` (Anchored) or `5` (Moored).
2.  `Speed Over Ground (SOG)` is consistently below **0.5 knots**.

These points are then clustered to form **Port Level Boundaries**, representing the jurisdictional and operational areas of maritime hubs.

---

## 2. Meso-scale: Non-Local Shipping Network (NLSN)
The NLSN scale identifies waypoint hubs—regions where vessels do not stop but exhibit distinct changes in their navigational behavior.

### Abrupt Trajectory Changes (Douglas-Peucker)
Instead of searching for low speed, NLSN focuses on **Dynamic Transitions**. We use the **Douglas-Peucker (DP)** algorithm to simplify vessel trajectories and extract "Feature Points".
*   **Waypoints**: Feature points represent "corners" in a trajectory where a vessel significantly changes its **Heading** or **Speed**.
*   **Restored Trajectories**: By connecting these feature points chronologically for each `trajectory_id`, we reconstruct the high-fidelity path segments.

These "abrupt change" regions are clustered to form **NLSN Hub Boundaries**, identifying critical maritime waypoints, choke points, and tactical maneuver areas.

---

## 3. Clustering Engine: CLIQUE
Both scales share a common clustering core based on the **CLIQUE (CLustering In QUEst)** algorithm.

### Grid-Based Subspace Clustering
CLIQUE is used because it handles massive AIS datasets efficiently by discretizing the geographical space into a grid.
*   **Hyperparameter $K$**: Defines the number of grid divisions (granularity).
*   **Hyperparameter $r$**: Defines the density threshold (minimum points/area) required to form a cluster.

### Boundary Extraction
Once clusters are identified, we use **Alpha-Shapes (Concave Hulls)** to generate tight-fitting GeoJSON boundaries around the point clouds. This differentiates "open sea" from "active maritime zones".

---

## 4. Integration & Visualization
The `IntegratedVisualizer` overlays these scales:
*   **Macro (Red)**: Large port boundaries and high-volume inter-port traffic.
*   **Meso (Green)**: Finer waypoint hubs and detailed trajectory segments (Restored Segments).
*   **Context (Heatmap)**: Background AIS density provides the "ground truth" traffic flow.

By adjusting the hyperparameters ($K$, $r$, $\gamma$), users can explore how the shipping network simplifies or complexifies at different resolutions.
