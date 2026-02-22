# Methodology: Multiscale Maritime Network Extraction

This document provides a technical and mathematical foundation for the extraction of maritime networks at three resolutions: **Macro (PLSN)**, **Meso (NLSN)**, and **Route (RLSN)**.

---

## 1. Macro-scale: Ports to Shipping Lanes (PLSN)
The PLSN scale identifies primary maritime infrastructure (ports) and the high-volume lanes connecting them.

### Stationary Point Detection (Anchoring & Mooring)
We define a point $p$ as **Stationary** ($P_S$) if it belongs to the subset of AIS reports where the vessel is not moving. Using AIS Message 1 (Dynamic) and Message 5 (Static/Voyage), we apply the following logic:

**Mathematical Definition:**
$$P_S = \{ p \in \text{AIS} \mid V_p < V_{threshold} \land S_p \in \{1, 5\} \}$$
Where:
- $V_p$ is the reported **Speed Over Ground (SOG)**.
- $V_{threshold}$ is typically $0.5$ knots.
- $S_p$ is the **Navigation Status** (1 = Anchored, 5 = Moored).

---

## 2. Meso-scale: Non-Local Shipping Network (NLSN)
The NLSN scale focuses on **Waypoints**—critical decision points where vessels change course or speed without stopping.

### Trajectory Simplification (Douglas-Peucker)
Individual vessel trajectories $T = \{c_1, c_2, ..., c_n\}$ are simplified using the **Douglas-Peucker (DP)** algorithm.

**Mechanism:**
For a segment between points $A$ and $B$, the algorithm finds the point $P$ with the maximum perpendicular distance $d$ from line segment $AB$.
- If $d > \epsilon$ (where $\epsilon$ is our **Gamma** parameter): The point $P$ is kept as a **Feature Point** (corner).
- If $d \le \epsilon$: All points between $A$ and $B$ are discarded.

**Mathematical Formalism:**
$$\Phi = \{ p \in T \mid \text{perpendicular\_distance}(p, \text{segment}(p_{prev}, p_{next})) > \epsilon \}$$

---

## 3. Route-scale: Route-Level Shipping Network (RLSN)
The RLSN provides the highest resolution, characterizing the statistical spread of traffic along established lanes.

### Slice-based Traffic Flow Fitting
For each edge between waypoints $A$ and $B$, we generate $M$ normal cross-sections (slices). AIS points $p$ are projected onto the normal vector $\vec{N}$ of the edge.

**Gaussian Probability Density Function (PDF):**
We fit the cross-sectional distribution using the Gaussian function:
$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
Where:
- $\mu$: The **Centroid** of the traffic flow (Customary Route).
- $\sigma$: The **Lateral Spread** (Navigational variance).

### Channel Boundary (3-Sigma Rule)
The spatial extent of the shipping lane is defined by the interval $[\mu - 3\sigma, \mu + 3\sigma]$. This encompasses 99.7% of all vessel transit behaviors, effectively identifying the "safe" navigational corridor.

---

## 4. Clustering Engine: CLIQUE (Subspace Clustering)
Both PLSN and NLSN scales utilize the **CLIQUE** algorithm for spatial partitioning.

**Formalism:**
1.  **Grid Partitioning**: Divide the 2D space (Lat/Lon) into $K \times K$ units.
2.  **Dense Unit Identification**: A unit $u$ is "dense" if its point count $C_u$ exceeds a threshold $\tau$:
    $$u \in \text{DenseUnits} \iff C_u > \tau$$
    Where $\tau = \text{density\_threshold\_r} \times N_{points}$.
3.  **Region Growing**: Connected dense units are merged into clusters $C_i$ using a graph-connectivity approach.
