from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RansacParameters:
    min_pt_num: int = 6
    iter_num: int = 2000
    th_dist: float = 0.1
    seed: int = 0


def depth_bilinear(points: np.ndarray, depth_img: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(2, 1)
    h, w = depth_img.shape
    x = pts[0]
    y = pts[1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    valid = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    out = np.full(pts.shape[1], 1e4, dtype=np.float64)
    if not valid.any():
        return out
    xv = x[valid]
    yv = y[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    ax = xv - x0v
    ay = yv - y0v
    out[valid] = (
        (1 - ax) * (1 - ay) * depth_img[y0v, x0v]
        + ax * (1 - ay) * depth_img[y0v, x0v + 1]
        + ax * ay * depth_img[y0v + 1, x0v + 1]
        + (1 - ax) * ay * depth_img[y0v + 1, x0v]
    )
    return out


def normalise2dpts(points_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points_h.astype(np.float64).copy()
    pts[:2] /= pts[2:3]
    centroid = pts[:2].mean(axis=1)
    shifted = pts[:2] - centroid[:, None]
    mean_dist = np.sqrt(np.sum(shifted * shifted, axis=0)).mean()
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 0 else 1.0
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return transform @ pts, transform


def _estimate_he_dlt(points1_xy_depth: np.ndarray, pts2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = points1_xy_depth.shape[1]
    a = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        x, y, z = points1_xy_depth[:, i]
        xp, yp = pts2[:, i]
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp, -xp, 1.0 / z, 0.0, -xp / z]
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp, -yp, 0.0, 1.0 / z, -yp / z]
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    x = vt[-1]
    return x[:9].reshape(3, 3), x[9:12]


def _map_points(pts: np.ndarray, depths: np.ndarray, h_inf: np.ndarray, e: np.ndarray) -> np.ndarray:
    homog = np.vstack([pts, np.ones(pts.shape[1], dtype=np.float64)])
    mapped_h = h_inf @ homog + e[:, None] / depths[None, :]
    return mapped_h[:2] / mapped_h[2:3]


def mapping_distances(pts1: np.ndarray, pts2: np.ndarray, depths1: np.ndarray, h_inf: np.ndarray, e: np.ndarray) -> np.ndarray:
    mapped = _map_points(pts1, depths1, h_inf, e)
    return np.sqrt(np.sum((mapped - pts2) ** 2, axis=0))


def depth_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    depth_img1: np.ndarray,
    params: RansacParameters = RansacParameters(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pts1.shape[1] < params.min_pt_num:
        raise ValueError(f"Need at least {params.min_pt_num} matches, got {pts1.shape[1]}")

    depth_pts1 = depth_bilinear(pts1, depth_img1)
    norm1, _ = normalise2dpts(np.vstack([pts1, np.ones(pts1.shape[1])]))
    norm2, _ = normalise2dpts(np.vstack([pts2, np.ones(pts2.shape[1])]))
    norm1_depth = np.vstack([norm1[:2], depth_pts1])
    norm2_xy = norm2[:2]

    rng = np.random.default_rng(params.seed)
    best_inliers = np.zeros(pts1.shape[1], dtype=bool)
    for _ in range(params.iter_num):
        sample = rng.choice(pts1.shape[1], size=params.min_pt_num, replace=False)
        try:
            h_norm, e_norm = _estimate_he_dlt(norm1_depth[:, sample], norm2_xy[:, sample])
            dists = mapping_distances(norm1[:2], norm2_xy, depth_pts1, h_norm, e_norm)
        except np.linalg.LinAlgError:
            continue
        inliers = dists < params.th_dist
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < params.min_pt_num:
        raise ValueError(f"Depth RANSAC found only {best_inliers.sum()} inliers")
    return pts1[:, best_inliers], pts2[:, best_inliers], best_inliers


def estimate_h_inf_and_epipole(pts1: np.ndarray, pts2: np.ndarray, depth_img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pts1.shape[1] < 6:
        raise ValueError(f"Need at least 6 inlier matches, got {pts1.shape[1]}")

    depth_pts1 = depth_bilinear(pts1, depth_img1)
    norm1, t1 = normalise2dpts(np.vstack([pts1, np.ones(pts1.shape[1])]))
    norm2, t2 = normalise2dpts(np.vstack([pts2, np.ones(pts2.shape[1])]))
    h_norm, e_norm = _estimate_he_dlt(np.vstack([norm1[:2], depth_pts1]), norm2[:2])
    h0 = np.linalg.solve(t2, h_norm @ t1)
    e0 = np.linalg.solve(t2, e_norm)
    x0 = np.concatenate([h0.reshape(-1), e0])

    def residual(x: np.ndarray) -> np.ndarray:
        h = x[:9].reshape(3, 3)
        e = x[9:12]
        mapped = _map_points(pts1, depth_pts1, h, e)
        return (mapped - pts2).reshape(-1)

    result = least_squares(residual, x0, method="lm", ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=1000)
    return result.x[:9].reshape(3, 3), result.x[9:12]
