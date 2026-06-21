from __future__ import annotations

import cv2
import numpy as np


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def sift_match(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int = 8000,
    ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps1, des1 = sift.detectAndCompute(_to_gray_u8(img1), None)
    kps2, des2 = sift.detectAndCompute(_to_gray_u8(img2), None)
    if des1 is None or des2 is None or len(kps1) < 2 or len(kps2) < 2:
        return np.empty((2, 0), dtype=np.float64), np.empty((2, 0), dtype=np.float64)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = matcher.knnMatch(des1, des2, k=2)
    matches = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            matches.append(m)
    matches.sort(key=lambda m: m.distance)

    seen1: set[tuple[int, int]] = set()
    seen2: set[tuple[int, int]] = set()
    pts1 = []
    pts2 = []
    for match in matches:
        p1 = kps1[match.queryIdx].pt
        p2 = kps2[match.trainIdx].pt
        key1 = (round(p1[0] * 1000), round(p1[1] * 1000))
        key2 = (round(p2[0] * 1000), round(p2[1] * 1000))
        if key1 in seen1 or key2 in seen2:
            continue
        seen1.add(key1)
        seen2.add(key2)
        pts1.append(p1)
        pts2.append(p2)

    if not pts1:
        return np.empty((2, 0), dtype=np.float64), np.empty((2, 0), dtype=np.float64)
    return np.asarray(pts1, dtype=np.float64).T, np.asarray(pts2, dtype=np.float64).T
