from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dsfn_py.inference import (  # noqa: E402
    load_fusion_network,
    load_warp_network,
    resolve_device,
    stitch_pair,
)
from dsfn_py.io_utils import save_rgb  # noqa: E402
from dsfn_py.run_stitchbench_general import default_fusion_ckpt, default_warp_ckpt  # noqa: E402

DEFAULT_RESULT_ROOT = r"D:\HD3D_Result"
DEFAULT_MANIFEST = r"D:\HD3D_Result\_work\manifest.json"
METHOD_FOLDER = "dsfn"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_datasets_file(result_root: Path) -> list[str]:
    datasets_path = result_root / "_work" / "datasets.txt"
    if not datasets_path.exists():
        return []
    return [
        line.strip()
        for line in datasets_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def write_method_status(
    method_dir: Path,
    *,
    pair_name: str,
    success: bool,
    runtime_seconds: float,
    raw_path: Path,
    failure_reason: str = "",
    error: str = "",
) -> None:
    payload = {
        "method": "DSFN",
        "pair_name": pair_name,
        "success": success,
        "runtime_seconds": runtime_seconds,
        "failure_reason": failure_reason,
        "raw_path": str(raw_path),
    }
    if error:
        payload["error"] = error
    method_dir.mkdir(parents=True, exist_ok=True)
    method_dir.joinpath("method_status.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _edge_attempts(max_input_edge: int) -> list[int]:
    if max_input_edge <= 0:
        return [0]
    edges = [max_input_edge]
    for edge in (1536, 1024, 768, 512, 384, 256):
        if edge < max_input_edge and edge not in edges:
            edges.append(edge)
    return edges


def release_memory(device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def process_entry(
    entry: dict[str, Any],
    result_root: Path,
    args: argparse.Namespace,
    warp_net,
    build_output_model,
    fusion_net,
    build_model,
    device,
) -> dict[str, Any]:
    pair_name = entry["pair_name"]
    method_dir = Path(entry["final_pair_dir"]) / METHOD_FOLDER
    raw_path = method_dir / "raw.png"
    left_path = Path(entry["left_source"])
    right_path = Path(entry["right_source"])

    if args.skip_existing and raw_path.exists():
        return {"pair_name": pair_name, "status": "skipped", "raw_path": str(raw_path)}

    started = time.perf_counter()
    last_error: BaseException | None = None
    for edge in _edge_attempts(args.max_input_edge):
        try:
            _, _, panorama = stitch_pair(
                left_path,
                right_path,
                warp_net,
                build_output_model,
                fusion_net,
                build_model,
                device,
                max_input_edge=edge,
            )
            save_rgb(raw_path, panorama)
            del panorama
            elapsed = time.perf_counter() - started
            write_method_status(
                method_dir,
                pair_name=pair_name,
                success=True,
                runtime_seconds=elapsed,
                raw_path=raw_path,
                failure_reason="",
            )
            return {
                "pair_name": pair_name,
                "status": "ok",
                "runtime_seconds": elapsed,
                "raw_path": str(raw_path),
                "max_input_edge": edge,
            }
        except Exception as exc:
            last_error = exc
            release_memory(device)
            continue

    elapsed = time.perf_counter() - started
    reason = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
    write_method_status(
        method_dir,
        pair_name=pair_name,
        success=False,
        runtime_seconds=elapsed,
        raw_path=raw_path,
        failure_reason=reason,
        error=traceback.format_exc() if last_error else "",
    )
    if last_error is not None:
        method_dir.joinpath("error.log").write_text(traceback.format_exc(), encoding="utf-8")
    return {"pair_name": pair_name, "status": "failed", "error": reason}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DSFN on HD3D manifest pairs.")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--warp-ckpt", default=str(default_warp_ckpt()))
    parser.add_argument("--fusion-ckpt", default=str(default_fusion_ckpt()))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--max-input-edge",
        type=int,
        default=1024,
        help="Initial longest-edge limit; auto-fallback to 768/512/384/256 on OOM.",
    )
    parser.add_argument("--pair", action="append", help="Limit to pair_name(s), e.g. Indoor_001_p12.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="Skip pairs that already have dsfn/raw.png.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_root = Path(args.result_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Run prepare_hd3d_pairs.py first."
        )

    manifest = load_manifest(manifest_path)
    if args.only_incomplete:
        manifest = [
            entry
            for entry in manifest
            if not (Path(entry["final_pair_dir"]) / METHOD_FOLDER / "raw.png").exists()
        ]
    if args.pair:
        wanted = set(args.pair)
        manifest = [entry for entry in manifest if entry["pair_name"] in wanted]
    elif load_datasets_file(result_root):
        order = load_datasets_file(result_root)
        by_name = {entry["pair_name"]: entry for entry in manifest}
        manifest = [by_name[name] for name in order if name in by_name]

    device = resolve_device(args.device)
    warp_net, build_output_model = load_warp_network(Path(args.warp_ckpt), device)
    fusion_net, build_model = load_fusion_network(Path(args.fusion_ckpt), device)

    ok_count = 0
    failed: list[dict[str, Any]] = []
    for entry in tqdm(manifest, desc="HD3D DSFN"):
        result = process_entry(
            entry,
            result_root,
            args,
            warp_net,
            build_output_model,
            fusion_net,
            build_model,
            device,
        )
        if result["status"] in ("ok", "skipped"):
            ok_count += 1
        else:
            failed.append(result)
            if args.stop_on_error:
                break
        release_memory(device)

    print(f"Finished {len(manifest)} pair(s): ok/skipped={ok_count}, failed={len(failed)}")
    if failed:
        for item in failed[:5]:
            print(f"  failed: {item['pair_name']} -> {item.get('error', '')}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
