from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def depth_png_name(image_path: str | Path) -> str:
    return f"{Path(image_path).stem}_depth_vitl.png"


def find_existing_depth(image_path: str | Path, *folders: str | Path | None) -> Path | None:
    name = depth_png_name(image_path)
    image_path = Path(image_path)
    candidates = [image_path.with_name(name)]
    for folder in folders:
        if folder is not None:
            candidates.append(Path(folder) / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def metric_depth_from_png(depth_png: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    depth_png = Path(depth_png)
    image = Image.open(depth_png)
    depth = np.asarray(image.convert("L"), dtype=np.float64)
    if size is not None:
        width, height = size
        if depth.shape[1] != width or depth.shape[0] != height:
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    depth = depth / 255.0
    depth = 1.0 / np.maximum(depth, 1e-6)
    return depth
