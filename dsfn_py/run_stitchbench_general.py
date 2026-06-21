from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .inference import load_fusion_network, load_warp_network, resolve_device, stitch_pair
from .io_utils import save_rgb

DEFAULT_MANIFEST = (
    r"C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss"
    r"\runs\depth_gsp_v5_planarity035\manifest.csv"
)


def default_warp_ckpt() -> Path:
    return Path(__file__).resolve().parents[1] / "Warp" / "model" / "epoch001_model_warp.pth"


def default_fusion_ckpt() -> Path:
    return Path(__file__).resolve().parents[1] / "Fusion" / "model" / "epoch002_model_fusion.pth"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_failed_runs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "category", "image1", "image2", "status", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_pair(
    manifest_row: dict[str, str],
    out_root: Path,
    args: argparse.Namespace,
    warp_net,
    build_output_model,
    fusion_net,
    build_model,
    device,
) -> dict[str, Any]:
    dataset = manifest_row["dataset"]
    category = manifest_row["category"]
    data_dir = Path(manifest_row["data_dir"])
    image_names = manifest_row["image_files"].split("|")[:2]
    if len(image_names) < 2:
        raise ValueError(f"Manifest row has fewer than two images: {dataset}")

    img_path1 = data_dir / image_names[0]
    img_path2 = data_dir / image_names[1]
    started = time.perf_counter()

    _, _, panorama = stitch_pair(
        img_path1,
        img_path2,
        warp_net,
        build_output_model,
        fusion_net,
        build_model,
        device,
        max_input_edge=args.max_input_edge,
    )

    out_dir = out_root / dataset
    save_rgb(out_dir / "panorama.png", panorama)
    elapsed = time.perf_counter() - started
    return {
        "dataset": dataset,
        "category": category,
        "image1": str(img_path1),
        "image2": str(img_path2),
        "status": "ok",
        "elapsed_sec": elapsed,
        "error": "",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DSFN on StitchBench General manifest pairs.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="OBJ-GSP manifest.csv with 100 General pairs.")
    parser.add_argument("--out", default="outputs/stitchbench_general", help="Output folder, ignored by git.")
    parser.add_argument("--warp-ckpt", default=str(default_warp_ckpt()), help="Warp checkpoint path.")
    parser.add_argument("--fusion-ckpt", default=str(default_fusion_ckpt()), help="Fusion checkpoint path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument(
        "--max-input-edge",
        type=int,
        default=2048,
        help="Resize inputs so the longest edge is at most this value. Use 0 to disable.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of manifest rows for smoke tests.")
    parser.add_argument("--dataset", action="append", default=None, help="Only run matching dataset name(s). Can be repeated.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest))
    if args.dataset:
        wanted = set(args.dataset)
        manifest = [row for row in manifest if row["dataset"] in wanted]
    if args.limit and args.limit > 0:
        manifest = manifest[: args.limit]

    device = resolve_device(args.device)
    warp_net, build_output_model = load_warp_network(Path(args.warp_ckpt), device)
    fusion_net, build_model = load_fusion_network(Path(args.fusion_ckpt), device)

    failed_rows: list[dict[str, Any]] = []
    ok_count = 0
    for row in tqdm(manifest, desc="StitchBench manifest"):
        try:
            process_pair(
                row,
                out_root,
                args,
                warp_net,
                build_output_model,
                fusion_net,
                build_model,
                device,
            )
            ok_count += 1
        except Exception as exc:
            image_names = row["image_files"].split("|")[:2]
            data_dir = Path(row["data_dir"])
            failed = {
                "dataset": row["dataset"],
                "category": row["category"],
                "image1": str(data_dir / image_names[0]) if image_names else "",
                "image2": str(data_dir / image_names[1]) if len(image_names) > 1 else "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failed_rows.append(failed)
            if args.stop_on_error:
                write_failed_runs(out_root / "failed_runs.csv", failed_rows)
                raise
    write_failed_runs(out_root / "failed_runs.csv", failed_rows)
    print(f"Wrote {ok_count}/{len(manifest)} panoramas to {out_root}")
    if failed_rows:
        print(f"Recorded {len(failed_rows)} failures in {out_root / 'failed_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
