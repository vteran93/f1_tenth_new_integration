import numpy as np
from typing import Tuple
from scipy.interpolate import splev, splrep

# --- Nearest point on trajectory (pure Python wrapper for Numba) ---
try:
    from numba import njit

    @njit(fastmath=False, cache=True)
    def _nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
        """
        Numba-accelerated projection of 'point' onto a piecewise linear 'trajectory'.
        Both inputs must have the same dtype.
        Returns: (proj_point, distance, t, segment_index)
        """
        # Compute segment differences
        diffs = trajectory[1:] - trajectory[:-1]
        l2s = diffs[:, 0]**2 + diffs[:, 1]**2
        # Dot products
        dots = np.empty(diffs.shape[0], dtype=trajectory.dtype)
        for i in range(dots.shape[0]):
            dots[i] = (point - trajectory[i]).dot(diffs[i])
        # Clamp t to [0,1]
        t = dots / l2s
        for i in range(t.shape[0]):
            if t[i] < 0.0:
                t[i] = 0.0
            elif t[i] > 1.0:
                t[i] = 1.0
        # Compute projections and distances
        projections = trajectory[:-1] + (t[:, None] * diffs)
        dists = np.sqrt(((point - projections)**2).sum(axis=1))
        idx = int(np.argmin(dists))
        return projections[idx], dists[idx], float(t[idx]), idx

except ImportError:
    # Fallback if Numba is unavailable
    def _nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
        diffs = trajectory[1:] - trajectory[:-1]
        l2s = (diffs**2).sum(axis=1)
        dots = np.einsum('j,ij->i', point - trajectory[:-1], diffs)
        t = np.clip(dots / l2s, 0.0, 1.0)
        projections = trajectory[:-1] + (t[:, None] * diffs)
        dists = np.linalg.norm(point - projections, axis=1)
        idx = int(np.argmin(dists))
        return projections[idx], dists[idx], float(t[idx]), idx


def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
    """
    Wrapper ensuring dtype compatibility before calling the numba-optimized function.
    """
    # Cast point to trajectory's dtype if mismatched
    if point.dtype != trajectory.dtype:
        point = point.astype(trajectory.dtype)
    return _nearest_point_on_trajectory(point, trajectory)

# --- Calculate curvatures via spline interpolation ---


def calculate_curvatures(points: np.ndarray) -> np.ndarray:
    """
    Calculate signed curvature for a sequence of points.
    Args:
        points: array of shape (N,2)
    Returns:
        curvatures: array of length N
    """
    x = points[:, 0]
    y = points[:, 1]
    t = np.arange(len(x), dtype=np.float64)
    spl_x = splrep(t, x.astype(np.float64))
    spl_y = splrep(t, y.astype(np.float64))
    dx = splev(t, spl_x, der=1)
    ddx = splev(t, spl_x, der=2)
    dy = splev(t, spl_y, der=1)
    ddy = splev(t, spl_y, der=2)
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-6)
    curv = (dx * ddy - dy * ddx) / denom
    return curv.astype(np.float32)
