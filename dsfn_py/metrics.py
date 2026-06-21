from __future__ import annotations

from functools import lru_cache

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def overlap_crop(target: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask_tar = target.sum(axis=2) > 0
    mask_ref = reference.sum(axis=2) > 0
    mask = mask_tar & mask_ref
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    y0, y1 = rows.min(), rows.max() + 1
    x0, x1 = cols.min(), cols.max() + 1
    return target[y0:y1, x0:x1], reference[y0:y1, x0:x1]


@lru_cache(maxsize=2)
def _lpips_model(device: str):
    import lpips

    model = lpips.LPIPS(net="alex")
    return model.to(device).eval()


def calculate_metrics(target: np.ndarray, reference: np.ndarray, lpips_device: str | None = None) -> dict[str, float | None]:
    crop = overlap_crop(target, reference)
    if crop is None:
        return {"psnr": None, "ssim": None, "lpips": None}
    target_crop, ref_crop = crop
    if min(target_crop.shape[:2]) < 7:
        return {"psnr": None, "ssim": None, "lpips": None}

    psnr = float(peak_signal_noise_ratio(ref_crop, target_crop, data_range=1.0))
    ssim = float(structural_similarity(ref_crop, target_crop, channel_axis=2, data_range=1.0))
    lpips_score = None
    if lpips_device is not None:
        try:
            import torch

            device = lpips_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = _lpips_model(device)
            with torch.no_grad():
                t = torch.from_numpy(target_crop.transpose(2, 0, 1)).float().unsqueeze(0) * 2.0 - 1.0
                r = torch.from_numpy(ref_crop.transpose(2, 0, 1)).float().unsqueeze(0) * 2.0 - 1.0
                lpips_score = float(model(t.to(device), r.to(device)).item())
        except Exception:
            lpips_score = None
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_score}
