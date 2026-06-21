from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: str | Path) -> list[object]:
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_images(folder: str | Path) -> list[Path]:
    return sorted(
        [
            p
            for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and "_depth_" not in p.stem.lower()
        ],
        key=natural_key,
    )


def iter_scene_dirs(dataset: str | Path) -> Iterable[Path]:
    dataset = Path(dataset)
    direct_images = list_images(dataset)
    if direct_images:
        yield dataset
        return
    yield from sorted([p for p in dataset.iterdir() if p.is_dir()], key=natural_key)


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    bgr = cv2.cvtColor((clipped * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
