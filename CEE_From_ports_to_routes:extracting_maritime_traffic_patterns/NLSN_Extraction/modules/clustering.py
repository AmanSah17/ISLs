import logging
import math
from collections import deque

import numpy as np
import pandas as pd
from numba import cuda


@cuda.jit
def _grid_index_kernel(lon, lat, lon_min, lon_span, lat_min, lat_span, k, out_x, out_y):
    i = cuda.grid(1)
    if i >= lon.shape[0]:
        return

    x = int((lon[i] - lon_min) / lon_span * k)
    y = int((lat[i] - lat_min) / lat_span * k)

    if x < 0:
        x = 0
    elif x >= k:
        x = k - 1

    if y < 0:
        y = 0
    elif y >= k:
        y = k - 1

    out_x[i] = x
    out_y[i] = y


class CLIQUEClusterer:
    """
    CLIQUE-style clustering for 2D geographic subspace (LON, LAT):
    1) Partition each dimension into K bins.
    2) Mark dense units using density threshold r.
    3) Identify connected dense units by DFS/BFS.
    """

    def __init__(
        self,
        k: int = 200,
        density_threshold_r: float = 0.0001,
        min_dense_points: int | None = None,
        neighbor_mode: str = "4",
        require_cuda: bool = True,
    ):
        self.k = int(k)
        self.density_threshold_r = float(density_threshold_r)
        self.min_dense_points = min_dense_points
        mode = str(neighbor_mode).strip()
        if mode.endswith(".0"):
            mode = mode[:-2]
        if mode not in {"4", "8"}:
            raise ValueError(f"neighbor_mode must be '4' or '8', got: {neighbor_mode}")
        self.neighbor_mode = mode
        self.require_cuda = bool(require_cuda)
        self.logger = logging.getLogger(__name__)

    def _assert_cuda_ready(self):
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is mandatory but no CUDA-capable GPU/runtime is available. "
                "Install compatible NVIDIA driver + CUDA toolkit and run in a CUDA-enabled environment."
            )

    def _compute_grid_indices_cuda(
        self,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        lon_min: float,
        lon_span: float,
        lat_min: float,
        lat_span: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._assert_cuda_ready()
        lon_arr = np.ascontiguousarray(lon_vals.astype(np.float32, copy=False))
        lat_arr = np.ascontiguousarray(lat_vals.astype(np.float32, copy=False))
        n = lon_arr.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        d_lon = cuda.to_device(lon_arr)
        d_lat = cuda.to_device(lat_arr)
        d_x = cuda.device_array(n, dtype=np.int32)
        d_y = cuda.device_array(n, dtype=np.int32)

        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        _grid_index_kernel[blocks_per_grid, threads_per_block](
            d_lon,
            d_lat,
            np.float32(lon_min),
            np.float32(lon_span),
            np.float32(lat_min),
            np.float32(lat_span),
            np.int32(self.k),
            d_x,
            d_y,
        )
        cuda.synchronize()
        return d_x.copy_to_host(), d_y.copy_to_host()

    def _compute_grid_indices_cpu(
        self,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        lon_min: float,
        lon_span: float,
        lat_min: float,
        lat_span: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lon_arr = np.asarray(lon_vals, dtype=np.float64)
        lat_arr = np.asarray(lat_vals, dtype=np.float64)
        if lon_arr.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        x = ((lon_arr - lon_min) / lon_span * self.k).astype(np.int32)
        y = ((lat_arr - lat_min) / lat_span * self.k).astype(np.int32)
        np.clip(x, 0, self.k - 1, out=x)
        np.clip(y, 0, self.k - 1, out=y)
        return x, y

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"Required column not found. Tried: {candidates}")

    def _get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        if self.neighbor_mode == "8":
            return [
                (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                (x - 1, y),                 (x + 1, y),
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
            ]
        return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(
            "Starting CLIQUE clustering (K=%d, r=%.8f, min_dense_points=%s, neighbor_mode=%s, require_cuda=%s)",
            self.k,
            self.density_threshold_r,
            str(self.min_dense_points),
            self.neighbor_mode,
            str(self.require_cuda),
        )

        if self.k < 2:
            raise ValueError("CLIQUE parameter K must be >= 2.")

        data = df.copy()
        lat_col = self._pick_column(data, ["LAT", "Latitude"])
        lon_col = self._pick_column(data, ["LON", "Longitude"])

        data[lat_col] = pd.to_numeric(data[lat_col], errors="coerce")
        data[lon_col] = pd.to_numeric(data[lon_col], errors="coerce")
        data = data[data[lat_col].notna() & data[lon_col].notna()].copy()

        n_points = len(data)
        if n_points == 0:
            self.logger.warning("No valid coordinate points available for clustering.")
            data["cluster_id"] = -1
            return data

        lon_min, lon_max = data[lon_col].min(), data[lon_col].max()
        lat_min, lat_max = data[lat_col].min(), data[lat_col].max()
        lon_span = max(lon_max - lon_min, 1e-12)
        lat_span = max(lat_max - lat_min, 1e-12)

        # Equal-width partitioning into K units on each dimension (paper parameter K).
        if self.require_cuda:
            clique_x, clique_y = self._compute_grid_indices_cuda(
                data[lon_col].to_numpy(),
                data[lat_col].to_numpy(),
                lon_min,
                lon_span,
                lat_min,
                lat_span,
            )
        else:
            self.logger.info("Using CPU grid indexing (require_cuda=False).")
            clique_x, clique_y = self._compute_grid_indices_cpu(
                data[lon_col].to_numpy(),
                data[lat_col].to_numpy(),
                lon_min,
                lon_span,
                lat_min,
                lat_span,
            )
        data["clique_x"] = clique_x
        data["clique_y"] = clique_y

        unit_counts = data.groupby(["clique_x", "clique_y"]).size().reset_index(name="count")

        if self.min_dense_points is not None:
            dense_threshold = int(self.min_dense_points)
        else:
            dense_threshold = max(1, int(math.ceil(self.density_threshold_r * n_points)))

        dense_units = unit_counts[unit_counts["count"] >= dense_threshold]
        self.logger.info(
            "Dense threshold=%d points per unit. Dense units found=%d/%d.",
            dense_threshold,
            len(dense_units),
            len(unit_counts),
        )

        if dense_units.empty:
            self.logger.warning("No dense units found with current CLIQUE parameters.")
            data["cluster_id"] = -1
            return data

        dense_set = set(zip(dense_units["clique_x"], dense_units["clique_y"]))
        visited = set()
        unit_to_cluster: dict[tuple[int, int], int] = {}
        cluster_id = 0

        # Connected-components over dense units (BFS/DFS equivalent).
        for start in dense_set:
            if start in visited:
                continue
            q = deque([start])
            visited.add(start)
            while q:
                ux, uy = q.popleft()
                unit_to_cluster[(ux, uy)] = cluster_id
                for nx, ny in self._get_neighbors(ux, uy):
                    candidate = (nx, ny)
                    if candidate in dense_set and candidate not in visited:
                        visited.add(candidate)
                        q.append(candidate)
            cluster_id += 1

        mapping_df = pd.DataFrame(
            [(ux, uy, cid) for (ux, uy), cid in unit_to_cluster.items()],
            columns=["clique_x", "clique_y", "cluster_id"],
        )
        result_df = data.merge(mapping_df, on=["clique_x", "clique_y"], how="left")
        result_df["cluster_id"] = result_df["cluster_id"].fillna(-1).astype(int)

        clustered_count = int((result_df["cluster_id"] != -1).sum())
        noise_count = int((result_df["cluster_id"] == -1).sum())
        self.logger.info(
            "Identified %d clusters. Clustered points=%d, noise=%d.",
            cluster_id,
            clustered_count,
            noise_count,
        )
        return result_df


class GridDensityClusterer(CLIQUEClusterer):
    """
    Backward-compatible alias.
    """

    def __init__(self, grid_size: float = 0.01, min_points: int = 10):
        # Approximate k from legacy grid size over 360 degrees for compatibility.
        k = max(2, int(round(360.0 / grid_size)))
        super().__init__(k=k, density_threshold_r=0.0, min_dense_points=min_points, neighbor_mode="8")
