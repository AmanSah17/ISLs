from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _to_points_array(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be shaped as (n, 2) for [lon, lat].")
    return arr


def polyline_length(points: np.ndarray) -> float:
    arr = _to_points_array(points)
    if arr.shape[0] < 2:
        return 0.0
    diffs = arr[1:] - arr[:-1]
    return float(np.sqrt((diffs * diffs).sum(axis=1)).sum())


def _point_to_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-20:
        delta = points - start
        return np.sqrt((delta * delta).sum(axis=1))

    rel = points - start
    t = np.clip((rel @ seg) / seg_len2, 0.0, 1.0)
    proj = start + np.outer(t, seg)
    delta = points - proj
    return np.sqrt((delta * delta).sum(axis=1))


@dataclass
class CompressionMetrics:
    gamma: float
    original_points: int
    compressed_points: int
    compression_rate_dr: float
    distance_similarity_dl: float
    distance_loss: float
    original_length: float
    compressed_length: float
    ld_score: float


@dataclass
class DPCompressionResult:
    keep_indices: np.ndarray
    compressed_points: np.ndarray
    metrics: CompressionMetrics


class DouglasPeucker:
    """
    Classic Douglas-Peucker polyline compression.
    """

    @staticmethod
    def compress_indices(points: np.ndarray, gamma: float) -> np.ndarray:
        arr = _to_points_array(points)
        n = arr.shape[0]
        if n <= 2:
            return np.arange(n, dtype=np.int64)

        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        keep[-1] = True

        stack: list[tuple[int, int]] = [(0, n - 1)]
        eps = float(max(gamma, 0.0))

        while stack:
            start_idx, end_idx = stack.pop()
            if end_idx <= start_idx + 1:
                continue

            segment_points = arr[start_idx + 1 : end_idx]
            distances = _point_to_segment_distances(
                segment_points,
                arr[start_idx],
                arr[end_idx],
            )

            if distances.size == 0:
                continue

            local_max_idx = int(np.argmax(distances))
            max_dist = float(distances[local_max_idx])
            if max_dist > eps:
                split_idx = start_idx + 1 + local_max_idx
                keep[split_idx] = True
                stack.append((start_idx, split_idx))
                stack.append((split_idx, end_idx))

        return np.flatnonzero(keep)

    @classmethod
    def compress(cls, points: np.ndarray, gamma: float) -> np.ndarray:
        arr = _to_points_array(points)
        keep_indices = cls.compress_indices(arr, gamma)
        return arr[keep_indices]


class AdaptiveDPOptimizer:
    """
    Adaptive DP utilities for feature point extraction.

    Dr = 1 - n/N
    Dl = 1 - |dist - dist_ori| / dist_ori
    LD = w1 * Dr + w2 * Dl

    gamma significance:
    - lower gamma: keeps more points (lower compression, higher geometry fidelity)
    - higher gamma: keeps fewer points (higher compression, potentially more distortion)
    """

    def __init__(self, w1: float = 1.0, w2: float = 1.0, eps: float = 1e-12):
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.eps = float(eps)

    def evaluate(self, original_points: np.ndarray, keep_indices: np.ndarray, gamma: float) -> CompressionMetrics:
        arr = _to_points_array(original_points)
        n_total = int(arr.shape[0])
        compressed = arr[keep_indices] if keep_indices.size > 0 else arr[:0]
        n_comp = int(compressed.shape[0])

        original_len = polyline_length(arr)
        compressed_len = polyline_length(compressed)
        distance_loss = abs(compressed_len - original_len)

        if n_total <= 0:
            dr = 0.0
        else:
            dr = 1.0 - (n_comp / n_total)

        if original_len <= self.eps:
            dl = 1.0
        else:
            dl = 1.0 - (distance_loss / (original_len + self.eps))
            dl = float(np.clip(dl, 0.0, 1.0))

        ld = float((self.w1 * dr) + (self.w2 * dl))

        return CompressionMetrics(
            gamma=float(gamma),
            original_points=n_total,
            compressed_points=n_comp,
            compression_rate_dr=float(dr),
            distance_similarity_dl=float(dl),
            distance_loss=float(distance_loss),
            original_length=float(original_len),
            compressed_length=float(compressed_len),
            ld_score=ld,
        )

    def compress_with_gamma(self, points: np.ndarray, gamma: float) -> DPCompressionResult:
        arr = _to_points_array(points)
        keep_indices = DouglasPeucker.compress_indices(arr, gamma=gamma)
        metrics = self.evaluate(arr, keep_indices, gamma=gamma)
        return DPCompressionResult(
            keep_indices=keep_indices,
            compressed_points=arr[keep_indices],
            metrics=metrics,
        )

    def compress_with_gamma_grid(self, points: np.ndarray, gamma_values: Iterable[float]) -> DPCompressionResult:
        arr = _to_points_array(points)
        best_result: DPCompressionResult | None = None
        for gamma in gamma_values:
            candidate = self.compress_with_gamma(arr, gamma=float(gamma))
            if best_result is None:
                best_result = candidate
                continue
            if candidate.metrics.ld_score > best_result.metrics.ld_score:
                best_result = candidate
            elif (
                candidate.metrics.ld_score == best_result.metrics.ld_score
                and candidate.metrics.compressed_points < best_result.metrics.compressed_points
            ):
                best_result = candidate
        if best_result is None:
            raise ValueError("gamma_values cannot be empty.")
        return best_result

    def compress_iterative(
        self,
        points: np.ndarray,
        gamma_start: float,
        gamma_step: float,
        min_gamma: float = 0.0,
        max_iter: int = 100,
    ) -> DPCompressionResult:
        """
        Algorithm-3 style adaptation:
        keep decreasing gamma while LD improves, and return previous best.
        """
        if gamma_step <= 0:
            raise ValueError("gamma_step must be > 0 for iterative adaptive DP.")

        arr = _to_points_array(points)
        gamma = float(gamma_start)
        best: DPCompressionResult | None = None
        previous_ld = float("-inf")

        for _ in range(max_iter):
            if gamma < min_gamma:
                break
            candidate = self.compress_with_gamma(arr, gamma=gamma)
            if candidate.metrics.ld_score >= previous_ld:
                best = candidate
                previous_ld = candidate.metrics.ld_score
                gamma -= gamma_step
            else:
                break

        if best is None:
            best = self.compress_with_gamma(arr, gamma=max(min_gamma, gamma_start))
        return best
