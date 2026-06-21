from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

GRID_H = 12
GRID_W = 12


@dataclass(frozen=True)
class WarpMeshes:
    src_vertices: np.ndarray
    dst_vertices: tuple[np.ndarray, np.ndarray]
    image_width: int
    image_height: int
    canvas_width: int
    canvas_height: int
    homography: np.ndarray
    grid_w: int = GRID_W
    grid_h: int = GRID_H


def homography_input_pixels_to_canvas(
    points_xy: np.ndarray,
    inverse_sampling_homography: np.ndarray,
    image_width: int,
    image_height: int,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    """Map input-image pixels to canvas pixels.

    DSFN's ``I_mat`` follows ``torch_homo_transform``: it maps normalized canvas
    coordinates to normalized input coordinates for inverse sampling. Canvas
    positions therefore use ``M @ inv(I_mat) @ N^{-1} @ pixel``.
    """
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    n_inv = np.array(
        [
            [2.0 / float(image_width), 0.0, -1.0],
            [0.0, 2.0 / float(image_height), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    m_mat = np.array(
        [
            [float(canvas_width) / 2.0, 0.0, float(canvas_width) / 2.0],
            [0.0, float(canvas_height) / 2.0, float(canvas_height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    forward = m_mat @ np.linalg.inv(inverse_sampling_homography) @ n_inv
    mapped = (np.concatenate([points_xy, np.ones((points_xy.shape[0], 1), dtype=np.float64)], axis=1) @ forward.T)
    return mapped[:, :2] / mapped[:, 2:3]


def _polygon_indices(nw: int, nh: int) -> list[list[int]]:
    polygons: list[list[int]] = []
    for h in range(nh):
        for w in range(nw):
            polygons.append(
                [
                    w + h * (nw + 1),
                    (w + 1) + h * (nw + 1),
                    (w + 1) + (h + 1) * (nw + 1),
                    w + (h + 1) * (nw + 1),
                ]
            )
    return polygons


def _grid_index(point: np.ndarray, image_width: int, image_height: int, nw: int, nh: int) -> int:
    lw = image_width / float(nw)
    lh = image_height / float(nh)
    gx = int(point[0] / lw)
    gy = int(point[1] / lh)
    if gx == nw:
        gx -= 1
    if gy == nh:
        gy -= 1
    gx = max(0, min(nw - 1, gx))
    gy = max(0, min(nh - 1, gy))
    return gx + gy * nw


def _verify_vertex_index(point: np.ndarray, polygon: list[int], vertices: np.ndarray) -> int:
    v1 = vertices[polygon[1]]
    v3 = vertices[polygon[3]]
    dist1 = float(np.sum((point - v1) ** 2))
    dist3 = float(np.sum((point - v3) ** 2))
    return polygon[3] if dist1 > dist3 else polygon[1]


def _affine_transform(
    src_vertices: np.ndarray,
    dst_vertices: np.ndarray,
    polygon: list[int],
    verify_index: int,
) -> np.ndarray:
    src_tri = np.float32(
        [
            src_vertices[polygon[0]],
            src_vertices[polygon[2]],
            src_vertices[verify_index],
        ]
    )
    dst_tri = np.float32(
        [
            dst_vertices[polygon[0]],
            dst_vertices[polygon[2]],
            dst_vertices[verify_index],
        ]
    )
    return cv2.getAffineTransform(src_tri, dst_tri)


def _apply_affine(affine: np.ndarray, point: np.ndarray) -> np.ndarray:
    homo = np.array([point[0], point[1], 1.0], dtype=np.float64)
    return affine @ homo


def compute_mesh_rmse(
    pts1: np.ndarray,
    pts2: np.ndarray,
    meshes: WarpMeshes,
) -> float:
    """Port of OBJ-GSP ``MultiImage::getRMSE`` on DSFN warp meshes."""
    nw, nh = meshes.grid_w, meshes.grid_h
    polygons = _polygon_indices(nw, nh)
    src_vertices = meshes.src_vertices
    dst_vertices1, dst_vertices2 = meshes.dst_vertices
    width = meshes.image_width
    height = meshes.image_height

    rmse_sum = 0.0
    feature_num = 0
    for index in range(pts1.shape[1]):
        p1 = pts1[:, index]
        p2 = pts2[:, index]
        idx1 = _grid_index(p1, width, height, nw, nh)
        idx2 = _grid_index(p2, width, height, nw, nh)
        if idx1 >= nh * nw or idx1 < 0 or idx2 >= nh * nw or idx2 < 0:
            continue

        poly1 = polygons[idx1]
        poly2 = polygons[idx2]
        verify1 = _verify_vertex_index(p1, poly1, src_vertices)
        verify2 = _verify_vertex_index(p2, poly2, src_vertices)

        affine1 = _affine_transform(src_vertices, dst_vertices1, poly1, verify1)
        affine2 = _affine_transform(src_vertices, dst_vertices2, poly2, verify2)
        warped1 = _apply_affine(affine1, p1)
        warped2 = _apply_affine(affine2, p2)
        rmse_sum += float(np.linalg.norm(warped1 - warped2))
        feature_num += 1

    if feature_num == 0:
        raise ValueError("No valid mesh cells for feature matches")
    return float(math.sqrt(rmse_sum / feature_num))


def _line_residual(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] < 2:
        return 0.0, 0.0
    points32 = np.asarray(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points32, cv2.DIST_L2, 0, 1e-2, 1e-2).reshape(-1)
    if abs(float(vx)) < 1e-12 and abs(float(vy)) < 1e-12:
        return 0.0, 0.0
    if abs(float(vx)) < 1e-12:
        a, b, c = 1.0, 0.0, -float(x0)
    else:
        a = float(vy / vx)
        c = float(y0 - a * x0)
        b = -1.0
    denom = math.hypot(a, b)
    residuals = np.abs(a * points[:, 0] + b * points[:, 1] + c) / denom
    avg = float(np.sqrt(np.mean(residuals * residuals)))
    sd = float(np.std(residuals))
    return avg, sd


def compute_warping_residual(meshes: WarpMeshes) -> tuple[float, float]:
    """Match OBJ-GSP MultiImage::getWarpingResidual averaged over both images."""
    nw, nh = meshes.grid_w, meshes.grid_h
    residual_avg = 0.0
    residual_sd = 0.0
    for dst_vertices in meshes.dst_vertices:
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for row_index in range(nh + 1):
            row_points = [dst_vertices[w + row_index * (nw + 1)] for w in range(nw + 1)]
            rows.append(np.asarray(row_points, dtype=np.float64))
        for col_index in range(nw + 1):
            col_points = [dst_vertices[col_index + row * (nw + 1)] for row in range(nh + 1)]
            cols.append(np.asarray(col_points, dtype=np.float64))

        sum_avg = 0.0
        sum_sd = 0.0
        for row_points in rows:
            avg, sd = _line_residual(row_points)
            sum_avg += avg
            sum_sd += sd
        for col_points in cols:
            avg, sd = _line_residual(col_points)
            sum_avg += avg
            sum_sd += sd
        residual_avg += sum_avg / (len(rows) + len(cols))
        residual_sd += sum_sd / (len(rows) + len(cols))

    count = len(meshes.dst_vertices)
    return residual_avg / count, residual_sd / count
