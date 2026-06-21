from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .mesh_rmse import WarpMeshes, homography_input_pixels_to_canvas


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_network(code_dir: Path):
    code_text = str(code_dir)
    if code_text in sys.path:
        sys.path.remove(code_text)
    sys.path.insert(0, code_text)
    importlib.invalidate_caches()
    if "network" in sys.modules:
        del sys.modules["network"]
    return importlib.import_module("network")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def bgr_to_input_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image = image_bgr.astype(np.float32)
    image = (image / 127.5) - 1.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def maybe_resize_pair(image1_bgr: np.ndarray, image2_bgr: np.ndarray, max_input_edge: int) -> tuple[np.ndarray, np.ndarray]:
    if max_input_edge <= 0:
        return image1_bgr, image2_bgr
    longest = max(image1_bgr.shape[0], image1_bgr.shape[1], image2_bgr.shape[0], image2_bgr.shape[1])
    if longest <= max_input_edge:
        return image1_bgr, image2_bgr
    scale = max_input_edge / float(longest)

    def _resize(image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

    return _resize(image1_bgr), _resize(image2_bgr)


def align_pair_shapes(image1_bgr: np.ndarray, image2_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(image1_bgr.shape[0], image2_bgr.shape[0])
    width = min(image1_bgr.shape[1], image2_bgr.shape[1])
    if image1_bgr.shape[:2] == (height, width) and image2_bgr.shape[:2] == (height, width):
        return image1_bgr, image2_bgr
    return (
        cv2.resize(image1_bgr, (width, height), interpolation=cv2.INTER_AREA),
        cv2.resize(image2_bgr, (width, height), interpolation=cv2.INTER_AREA),
    )


def tensor_to_rgb01(tensor: torch.Tensor) -> np.ndarray:
    image = ((tensor[0].detach().cpu().numpy() + 1.0) * 127.5).transpose(1, 2, 0)
    return np.clip(image, 0.0, 255.0).astype(np.float64) / 255.0


def mask_tensor_to_single_channel(mask_tensor: torch.Tensor) -> torch.Tensor:
    if mask_tensor.shape[1] == 1:
        return mask_tensor
    return mask_tensor[:, 0:1, ...]


# Fusion U-Net pools 4x; down5 uses dilation=5 (effective 11x11 kernel).
MIN_FUSION_EDGE = 176


def _pad_spatial_tensor(tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    _, _, height, width = tensor.shape
    if height >= target_height and width >= target_width:
        return tensor
    pad_bottom = max(0, target_height - height)
    pad_right = max(0, target_width - width)
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")


def pad_for_fusion(
    warp1: torch.Tensor,
    warp2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    min_edge: int = MIN_FUSION_EDGE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    height = max(warp1.shape[2], warp2.shape[2], mask1.shape[2], mask2.shape[2])
    width = max(warp1.shape[3], warp2.shape[3], mask1.shape[3], mask2.shape[3])
    target_height = max(height, min_edge)
    target_width = max(width, min_edge)
    return (
        _pad_spatial_tensor(warp1, target_height, target_width),
        _pad_spatial_tensor(warp2, target_height, target_width),
        _pad_spatial_tensor(mask1, target_height, target_width),
        _pad_spatial_tensor(mask2, target_height, target_width),
        (height, width),
    )


def _crop_spatial_tensor(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tensor[:, :, :height, :width]


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "not enough memory" in message
        or "unable to allocate" in message
        or "defaultcpuallocator" in message
    )


def _fallback_input_edges(max_input_edge: int) -> list[int]:
    if max_input_edge <= 0:
        return [0]
    edges = [max_input_edge]
    for edge in (1536, 1024, 768, 512, 384, 256):
        if edge < max_input_edge and edge not in edges:
            edges.append(edge)
    return edges


def _is_fusion_size_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "kernel size can't be greater than actual input size" in message


def _fusion_scale_factors() -> tuple[float, ...]:
    return (1.0, 1.5, 2.0, 3.0)


def _run_fusion(
    fusion_net,
    build_model,
    warp1: torch.Tensor,
    warp2: torch.Tensor,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:
    crop_height = warp1.shape[2]
    crop_width = warp1.shape[3]
    last_error: BaseException | None = None
    for scale in _fusion_scale_factors():
        if scale == 1.0:
            fw1, fw2, fm1, fm2 = warp1, warp2, mask1, mask2
        else:
            scaled_height = max(MIN_FUSION_EDGE, int(round(crop_height * scale)))
            scaled_width = max(MIN_FUSION_EDGE, int(round(crop_width * scale)))
            size = (scaled_height, scaled_width)
            fw1 = F.interpolate(warp1, size=size, mode="bilinear", align_corners=False)
            fw2 = F.interpolate(warp2, size=size, mode="bilinear", align_corners=False)
            fm1 = F.interpolate(mask1, size=size, mode="bilinear", align_corners=False)
            fm2 = F.interpolate(mask2, size=size, mode="bilinear", align_corners=False)
        fw1, fw2, fm1, fm2, (fusion_height, fusion_width) = pad_for_fusion(fw1, fw2, fm1, fm2)
        try:
            fusion_out = build_model(fusion_net, fw1, fw2, fm1, fm2)
            stitched = _crop_spatial_tensor(fusion_out["stitched_image"], fusion_height, fusion_width)
            if scale != 1.0:
                stitched = F.interpolate(
                    stitched,
                    size=(crop_height, crop_width),
                    mode="bilinear",
                    align_corners=False,
                )
            return stitched
        except RuntimeError as exc:
            last_error = exc
            if not _is_fusion_size_error(exc):
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("fusion failed without raising an exception")


def load_warp_network(checkpoint_path: Path, device: torch.device):
    warp_dir = _repo_root() / "Warp" / "newCodes"
    warp_network = _import_network(warp_dir)
    net = warp_network.Network().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net, warp_network.build_output_model


def load_fusion_network(checkpoint_path: Path, device: torch.device):
    fusion_dir = _repo_root() / "Fusion" / "newCodes"
    fusion_network = _import_network(fusion_dir)
    net = fusion_network.Network().to(device)
    for module in net.modules():
        if isinstance(module, fusion_network.RepConvN):
            module.fuse_convs()
            module.forward = module.forward_fuse
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net, fusion_network.build_model


def prepare_aligned_stitch_pair(
    image1_bgr: np.ndarray,
    image2_bgr: np.ndarray,
    max_input_edge: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    resized1, resized2 = maybe_resize_pair(image1_bgr, image2_bgr, max_input_edge)
    return align_pair_shapes(resized1, resized2)


def bgr_to_rgb01(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def _mesh_vertices_from_canvas_homography(
    rigid_mesh: torch.Tensor,
    inverse_sampling_homography: torch.Tensor,
    image_width: int,
    image_height: int,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    points = rigid_mesh[0].reshape(-1, 2).detach().cpu().numpy()
    homography = inverse_sampling_homography[0].detach().cpu().numpy()
    return homography_input_pixels_to_canvas(
        points,
        homography,
        image_width,
        image_height,
        canvas_width,
        canvas_height,
    )


def _warp_meshes_from_tensors(warp_out: dict, image_width: int, image_height: int) -> WarpMeshes:
    canvas_height, canvas_width = warp_out["final_warp1"].shape[2:4]
    src_vertices = warp_out["mesh1"][0].detach().cpu().numpy().reshape(-1, 2)
    dst_vertices1 = _mesh_vertices_from_canvas_homography(
        warp_out["mesh1"],
        warp_out["I_mat"],
        image_width,
        image_height,
        int(canvas_width),
        int(canvas_height),
    )
    dst_vertices2 = warp_out["mesh2"][0].detach().cpu().numpy().reshape(-1, 2)
    homography = warp_out["I_mat"][0].detach().cpu().numpy()
    return WarpMeshes(
        src_vertices=src_vertices,
        dst_vertices=(dst_vertices1, dst_vertices2),
        image_width=image_width,
        image_height=image_height,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        homography=homography,
    )


def compute_warp_meshes_from_aligned(
    aligned1_bgr: np.ndarray,
    aligned2_bgr: np.ndarray,
    warp_net,
    build_output_model,
    device: torch.device,
) -> WarpMeshes:
    height, width = aligned1_bgr.shape[:2]
    with torch.no_grad():
        input1 = bgr_to_input_tensor(aligned1_bgr, device)
        input2 = bgr_to_input_tensor(aligned2_bgr, device)
        warp_out = build_output_model(warp_net, input1, input2)
    return _warp_meshes_from_tensors(warp_out, width, height)


def compute_warp_meshes(
    image1: np.ndarray,
    image2: np.ndarray,
    warp_net,
    build_output_model,
    device: torch.device,
    max_input_edge: int = 0,
) -> WarpMeshes:
    last_error: BaseException | None = None
    try:
        for edge in _fallback_input_edges(max_input_edge):
            aligned1, aligned2 = prepare_aligned_stitch_pair(image1, image2, edge)
            try:
                return compute_warp_meshes_from_aligned(
                    aligned1,
                    aligned2,
                    warp_net,
                    build_output_model,
                    device,
                )
            except Exception as exc:
                last_error = exc
                if not _is_oom_error(exc):
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if last_error is not None:
            raise last_error
        raise RuntimeError("compute_warp_meshes failed without raising an exception")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _stitch_pair_once(
    image1: np.ndarray,
    image2: np.ndarray,
    warp_net,
    build_output_model,
    fusion_net,
    build_model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input1 = bgr_to_input_tensor(image1, device)
    input2 = bgr_to_input_tensor(image2, device)

    with torch.no_grad():
        warp_out = build_output_model(warp_net, input1, input2)
        warp1 = warp_out["final_warp1"]
        warp2 = warp_out["final_warp2"]
        mask1 = mask_tensor_to_single_channel(warp_out["final_warp1_mask"])
        mask2 = mask_tensor_to_single_channel(warp_out["final_warp2_mask"])
        stitched = _run_fusion(fusion_net, build_model, warp1, warp2, mask1, mask2)

    warped_reference = tensor_to_rgb01(warp_out["final_warp1"])
    warped_target = tensor_to_rgb01(warp_out["final_warp2"])
    panorama = tensor_to_rgb01(stitched)
    return warped_reference, warped_target, panorama


def stitch_pair(
    image_path1: Path,
    image_path2: Path,
    warp_net,
    build_output_model,
    fusion_net,
    build_model,
    device: torch.device,
    max_input_edge: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image1 = cv2.imread(str(image_path1), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(image_path2), cv2.IMREAD_COLOR)
    if image1 is None or image2 is None:
        raise FileNotFoundError(f"Could not read image pair: {image_path1}, {image_path2}")

    last_error: BaseException | None = None
    try:
        for edge in _fallback_input_edges(max_input_edge):
            resized1, resized2 = maybe_resize_pair(image1, image2, edge)
            aligned1, aligned2 = align_pair_shapes(resized1, resized2)
            try:
                return _stitch_pair_once(
                    aligned1,
                    aligned2,
                    warp_net,
                    build_output_model,
                    fusion_net,
                    build_model,
                    device,
                )
            except Exception as exc:
                last_error = exc
                if not _is_oom_error(exc):
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if last_error is not None:
            raise last_error
        raise RuntimeError("stitch_pair failed without raising an exception")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
