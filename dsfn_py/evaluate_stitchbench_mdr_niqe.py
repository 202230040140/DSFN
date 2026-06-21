from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .features import sift_match
from .inference import (
    _fallback_input_edges,
    _is_oom_error,
    bgr_to_rgb01,
    compute_warp_meshes_from_aligned,
    load_warp_network,
    prepare_aligned_stitch_pair,
    resolve_device,
)
from .io_utils import read_rgb
from .mesh_rmse import compute_mesh_rmse, compute_warping_residual
from .run_stitchbench_general import default_warp_ckpt

DEFAULT_MANIFEST = (
    r"C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss"
    r"\runs\depth_gsp_v5_planarity035\manifest.csv"
)
DEFAULT_DEPTH_GSP_ROOT = (
    r"C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss"
    r"\runs\depth_gsp_v5_planarity035"
)

CATEGORIES = ("OBJ-GSP", "AANAP", "APAP", "CAVE", "DFW", "DHW", "GES", "LPC", "REW", "SEAGULL", "SVA", "SPHP")

PAPER_TARGETS = {
    "OBJ-GSP": {"mdr": 1.12229, "niqe": 2.54906},
    "AANAP": {"mdr": 1.05930, "niqe": 2.74965},
    "APAP": {"mdr": 1.20123, "niqe": 3.39280},
    "CAVE": {"mdr": 0.89731, "niqe": 4.01565},
    "DFW": {"mdr": 0.97259, "niqe": 5.69104},
    "DHW": {"mdr": 1.00496, "niqe": 2.60825},
    "GES": {"mdr": 0.98288, "niqe": 3.70041},
    "LPC": {"mdr": 1.10622, "niqe": 3.23057},
    "REW": {"mdr": 1.08635, "niqe": 2.81480},
    "SEAGULL": {"mdr": 1.08296, "niqe": 4.08903},
    "SVA": {"mdr": 1.47813, "niqe": 6.96149},
    "SPHP": {"mdr": 1.07699, "niqe": 2.49712},
}

PER_PAIR_COLUMNS = (
    "dataset",
    "category",
    "result_image",
    "mdr_rmse",
    "warping_residual_avg",
    "warping_residual_sd",
    "niqe",
    "status",
)


def parse_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def pct(value: float) -> str:
    return "" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def rel_gap(candidate_value: float, baseline_value: float) -> float:
    if not math.isfinite(candidate_value) or not math.isfinite(baseline_value) or baseline_value == 0:
        return math.nan
    return candidate_value / baseline_value - 1.0


def ratio(candidate_value: float, baseline_value: float) -> float:
    if not math.isfinite(candidate_value) or not math.isfinite(baseline_value) or baseline_value == 0:
        return math.nan
    return candidate_value / baseline_value


def pass_status(value: float, target: float, *, allow_abs: Optional[float] = None) -> str:
    if not math.isfinite(value):
        return "Missing"
    relative_ok = abs(value - target) / target <= 0.15
    absolute_ok = allow_abs is not None and abs(value - target) <= allow_abs
    return "Pass" if relative_ok or absolute_ok else "Needs Review"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_per_pair(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["mdr_rmse"] = parse_float(row.get("mdr_rmse"))
            row["warping_residual_avg"] = parse_float(row.get("warping_residual_avg"))
            row["warping_residual_sd"] = parse_float(row.get("warping_residual_sd"))
            row["niqe"] = parse_float(row.get("niqe"))
            rows[row["dataset"]] = row
    return rows


def is_ok(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("status") == "ok"
        and math.isfinite(parse_float(row.get("mdr_rmse")))
        and math.isfinite(parse_float(row.get("niqe")))
    )


def load_niqe_metric(device: str):
    import pyiqa

    return pyiqa.create_metric("niqe", device=device)


def compute_niqe(metric, image_path: Path) -> float:
    if not image_path.exists():
        return math.nan
    try:
        score = metric(str(image_path))
    except Exception:
        import torch

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return math.nan
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        score = metric(tensor)
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def compute_mesh_mdr(
    manifest_row: dict[str, str],
    warp_net,
    build_output_model,
    device,
    max_input_edge: int,
) -> tuple[float, float, float]:
    image_names = manifest_row["image_files"].split("|")[:2]
    image_paths = [Path(manifest_row["data_dir"]) / name for name in image_names]
    img1 = read_rgb(image_paths[0])
    img2 = read_rgb(image_paths[1])
    image1_bgr = cv2.cvtColor((img1 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image2_bgr = cv2.cvtColor((img2 * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    last_error: BaseException | None = None
    try:
        for edge in _fallback_input_edges(max_input_edge):
            aligned1, aligned2 = prepare_aligned_stitch_pair(image1_bgr, image2_bgr, edge)
            try:
                meshes = compute_warp_meshes_from_aligned(
                    aligned1,
                    aligned2,
                    warp_net,
                    build_output_model,
                    device,
                )
                pts1, pts2 = sift_match(bgr_to_rgb01(aligned1), bgr_to_rgb01(aligned2))
                mdr = compute_mesh_rmse(pts1, pts2, meshes)
                residual_avg, residual_sd = compute_warping_residual(meshes)
                return mdr, residual_avg, residual_sd
            except Exception as exc:
                last_error = exc
                if not _is_oom_error(exc):
                    raise
                if device.type == "cuda":
                    import torch

                    torch.cuda.empty_cache()
        if last_error is not None:
            raise last_error
        raise RuntimeError("compute_mesh_mdr failed without raising an exception")
    finally:
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()


def evaluate_dsfn(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_manifest(Path(args.manifest))
    dsfn_root = Path(args.dsfn_root)
    metric = None if args.skip_niqe else load_niqe_metric(args.device)
    device = resolve_device(args.device)
    warp_net = build_output_model = None
    if not args.skip_mdr:
        warp_net, build_output_model = load_warp_network(Path(args.warp_ckpt), device)
    rows: list[dict[str, Any]] = []

    for row in tqdm(manifest, desc="DSFN MDR/NIQE"):
        dataset = row["dataset"]
        category = row["category"]
        panorama = dsfn_root / dataset / "panorama.png"
        mdr = residual_avg = residual_sd = niqe = math.nan
        if not panorama.exists():
            status = "missing_result"
        else:
            status = "ok"
            if not args.skip_mdr:
                try:
                    mdr, residual_avg, residual_sd = compute_mesh_mdr(
                        row,
                        warp_net,
                        build_output_model,
                        device,
                        args.max_input_edge,
                    )
                except Exception:
                    status = "failed"
            if metric is not None:
                try:
                    niqe = compute_niqe(metric, panorama)
                except Exception:
                    status = "failed"
        rows.append(
            {
                "dataset": dataset,
                "category": category,
                "result_image": str(panorama),
                "mdr_rmse": mdr,
                "warping_residual_avg": residual_avg,
                "warping_residual_sd": residual_sd,
                "niqe": niqe,
                "status": status,
            }
        )
    return rows


def write_by_category(output_root: Path, rows: list[dict[str, Any]]) -> None:
    category_rows = []
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        mdr_values = [parse_float(row["mdr_rmse"]) for row in group]
        niqe_values = [parse_float(row["niqe"]) for row in group]
        category_rows.append(
            {
                "category": category,
                "total_count": len(group),
                "valid_mdr_count": len([value for value in mdr_values if math.isfinite(value)]),
                "valid_niqe_count": len([value for value in niqe_values if math.isfinite(value)]),
                "mdr_rmse_mean": finite_mean(mdr_values),
                "warping_residual_avg_mean": finite_mean([parse_float(row["warping_residual_avg"]) for row in group]),
                "warping_residual_sd_mean": finite_mean([parse_float(row["warping_residual_sd"]) for row in group]),
                "niqe_mean": finite_mean(niqe_values),
            }
        )
    write_csv(output_root / "by_category.csv", category_rows)


def write_paper_comparison(output_root: Path, rows: list[dict[str, Any]]) -> None:
    comparison_rows = []
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        mdr_values = [parse_float(row["mdr_rmse"]) for row in group]
        niqe_values = [parse_float(row["niqe"]) for row in group]
        valid_mdr_count = len([value for value in mdr_values if math.isfinite(value)])
        valid_niqe_count = len([value for value in niqe_values if math.isfinite(value)])
        mdr_mean = finite_mean(mdr_values)
        niqe_mean = finite_mean(niqe_values)
        target = PAPER_TARGETS[category]
        mdr_status = pass_status(mdr_mean, target["mdr"])
        niqe_status = pass_status(niqe_mean, target["niqe"], allow_abs=0.5)
        overall = "Pass" if mdr_status == "Pass" and niqe_status == "Pass" else "Needs Review"
        if mdr_status == "Missing" or niqe_status == "Missing":
            overall = "Missing"
        comparison_rows.append(
            {
                "category": category,
                "total_count": len(group),
                "valid_mdr_count": valid_mdr_count,
                "valid_niqe_count": valid_niqe_count,
                "paper_mdr": target["mdr"],
                "ours_mdr": mdr_mean,
                "mdr_relative_error": abs(mdr_mean - target["mdr"]) / target["mdr"] if math.isfinite(mdr_mean) else math.nan,
                "mdr_status": mdr_status,
                "paper_niqe": target["niqe"],
                "ours_niqe": niqe_mean,
                "niqe_relative_error": abs(niqe_mean - target["niqe"]) / target["niqe"] if math.isfinite(niqe_mean) else math.nan,
                "niqe_abs_error": abs(niqe_mean - target["niqe"]) if math.isfinite(niqe_mean) else math.nan,
                "niqe_status": niqe_status,
                "overall_status": overall,
            }
        )
    write_csv(output_root / "paper_comparison.csv", comparison_rows)
    return comparison_rows


def write_report(output_root: Path, comparison_rows: list[dict[str, Any]]) -> None:
    report_lines = [
        "# DSFN StitchBench General Report",
        "",
        "Result image: `panorama.png`",
        "",
        "| Category | Valid/Total | Paper MDR | Ours MDR | MDR Status | Paper NIQE | Ours NIQE | NIQE Status | Overall |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in comparison_rows:
        report_lines.append(
            "| {category} | {valid_count}/{total_count} | {paper_mdr:.5f} | {ours_mdr} | {mdr_status} | {paper_niqe:.5f} | {ours_niqe} | {niqe_status} | {overall_status} |".format(
                category=row["category"],
                valid_count=row["valid_mdr_count"],
                total_count=row["total_count"],
                paper_mdr=row["paper_mdr"],
                ours_mdr=fmt(row["ours_mdr"]),
                mdr_status=row["mdr_status"],
                paper_niqe=row["paper_niqe"],
                ours_niqe=fmt(row["ours_niqe"]),
                niqe_status=row["niqe_status"],
                overall_status=row["overall_status"],
            )
        )
    report_lines.extend(
        [
            "",
        "MDR note: DSFN `mdr_rmse` uses the OBJ-GSP mesh RMSE formula (`MultiImage::getRMSE`) on DSFN warp meshes with SIFT correspondences.",
        "NIQE is computed on `panorama.png` with pyiqa.",
        "Depth-GSP baseline MDR is read from C++ debug `{dataset}-RMSE-[DPS].txt`; feature sets may differ slightly from SIFT.",
            "Paper MDR/NIQE targets refer to the published OBJ-GSP row for each category.",
        ]
    )
    (output_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def compare_to_depth_gsp(output_root: Path, candidate_rows: list[dict[str, Any]], baseline_root: Path) -> None:
    baseline = load_per_pair(baseline_root / "per_pair.csv")
    candidate = {row["dataset"]: row for row in candidate_rows}
    names = sorted(set(baseline) | set(candidate))
    rows = []
    common_ok = []
    baseline_failed_candidate_ok = []
    candidate_failed = []
    for name in names:
        base = baseline.get(name)
        cand = candidate.get(name)
        base_ok = is_ok(base)
        cand_ok = is_ok(cand)
        if base_ok and cand_ok:
            common_ok.append(name)
        elif not base_ok and cand_ok:
            baseline_failed_candidate_ok.append(name)
        elif not cand_ok:
            candidate_failed.append(name)

        category = (cand or base or {}).get("category", "")
        base_mdr = parse_float(base.get("mdr_rmse")) if base else math.nan
        cand_mdr = parse_float(cand.get("mdr_rmse")) if cand else math.nan
        base_niqe = parse_float(base.get("niqe")) if base else math.nan
        cand_niqe = parse_float(cand.get("niqe")) if cand else math.nan
        mdr_delta = cand_mdr - base_mdr if base_ok and cand_ok else math.nan
        niqe_delta = cand_niqe - base_niqe if base_ok and cand_ok else math.nan
        rows.append(
            {
                "dataset": name,
                "category": category,
                "baseline_status": base.get("status", "missing") if base else "missing",
                "candidate_status": cand.get("status", "missing") if cand else "missing",
                "baseline_mdr": base_mdr,
                "candidate_mdr": cand_mdr,
                "mdr_delta": mdr_delta,
                "mdr_rel_gap": rel_gap(cand_mdr, base_mdr) if base_ok and cand_ok else math.nan,
                "mdr_ratio": ratio(cand_mdr, base_mdr) if base_ok and cand_ok else math.nan,
                "mdr_better": int(mdr_delta < 0) if math.isfinite(mdr_delta) else "",
                "baseline_niqe": base_niqe,
                "candidate_niqe": cand_niqe,
                "niqe_delta": niqe_delta,
                "niqe_rel_gap": rel_gap(cand_niqe, base_niqe) if base_ok and cand_ok else math.nan,
                "niqe_ratio": ratio(cand_niqe, base_niqe) if base_ok and cand_ok else math.nan,
                "niqe_better": int(niqe_delta < 0) if math.isfinite(niqe_delta) else "",
                "both_better": int(mdr_delta < 0 and niqe_delta < 0) if math.isfinite(mdr_delta) and math.isfinite(niqe_delta) else "",
                "candidate_result_image": cand.get("result_image", "") if cand else "",
                "baseline_result_image": base.get("result_image", "") if base else "",
            }
        )
    write_csv(output_root / "method_pair_comparison.csv", rows)

    by_category: dict[str, list[str]] = defaultdict(list)
    for name in common_ok:
        by_category[(candidate.get(name) or baseline.get(name)).get("category", "")].append(name)
    category_rows = []
    for category in sorted(by_category):
        category_names = by_category[category]
        both_better = [
            name
            for name in category_names
            if parse_float(candidate[name]["mdr_rmse"]) < parse_float(baseline[name]["mdr_rmse"])
            and parse_float(candidate[name]["niqe"]) < parse_float(baseline[name]["niqe"])
        ]
        category_rows.append(
            {
                "category": category,
                "common_ok_count": len(category_names),
                "candidate_mdr_mean": finite_mean([parse_float(candidate[name]["mdr_rmse"]) for name in category_names]),
                "baseline_mdr_mean": finite_mean([parse_float(baseline[name]["mdr_rmse"]) for name in category_names]),
                "candidate_niqe_mean": finite_mean([parse_float(candidate[name]["niqe"]) for name in category_names]),
                "baseline_niqe_mean": finite_mean([parse_float(baseline[name]["niqe"]) for name in category_names]),
                "both_better_count": len(both_better),
                "both_better_rate": len(both_better) / len(category_names),
            }
        )
    write_csv(output_root / "method_category_comparison.csv", category_rows)

    common_candidate_mdr = finite_mean([parse_float(candidate[name]["mdr_rmse"]) for name in common_ok])
    common_baseline_mdr = finite_mean([parse_float(baseline[name]["mdr_rmse"]) for name in common_ok])
    common_candidate_niqe = finite_mean([parse_float(candidate[name]["niqe"]) for name in common_ok])
    common_baseline_niqe = finite_mean([parse_float(baseline[name]["niqe"]) for name in common_ok])
    mdr_better = [name for name in common_ok if parse_float(candidate[name]["mdr_rmse"]) < parse_float(baseline[name]["mdr_rmse"])]
    niqe_better = [name for name in common_ok if parse_float(candidate[name]["niqe"]) < parse_float(baseline[name]["niqe"])]
    both_better = [
        name
        for name in common_ok
        if parse_float(candidate[name]["mdr_rmse"]) < parse_float(baseline[name]["mdr_rmse"])
        and parse_float(candidate[name]["niqe"]) < parse_float(baseline[name]["niqe"])
    ]

    report = [
        "# DSFN vs Depth-GSP-v5",
        "",
        "## Summary",
        "",
        f"- Total datasets: {len(names)}",
        f"- Common successful datasets: {len(common_ok)}",
        f"- Depth-GSP-v5 failed while DSFN succeeded: {len(baseline_failed_candidate_ok)}",
        f"- DSFN failed: {len(candidate_failed)}",
        f"- Common mean MDR/RMSE: DSFN {fmt(common_candidate_mdr)} vs Depth-GSP-v5 {fmt(common_baseline_mdr)}",
        f"- Common mean MDR/RMSE relative gap: {pct(rel_gap(common_candidate_mdr, common_baseline_mdr))}",
        f"- Common mean NIQE: DSFN {fmt(common_candidate_niqe)} vs Depth-GSP-v5 {fmt(common_baseline_niqe)}",
        f"- Common mean NIQE relative gap: {pct(rel_gap(common_candidate_niqe, common_baseline_niqe))}",
        f"- MDR/RMSE better on common set: {len(mdr_better)}/{len(common_ok)} ({pct(len(mdr_better) / len(common_ok) if common_ok else math.nan)})",
        f"- NIQE better on common set: {len(niqe_better)}/{len(common_ok)} ({pct(len(niqe_better) / len(common_ok) if common_ok else math.nan)})",
        f"- Both MDR/RMSE and NIQE better on common set: {len(both_better)}/{len(common_ok)} ({pct(len(both_better) / len(common_ok) if common_ok else math.nan)})",
        "",
        "Metric note: Both methods report OBJ-GSP mesh RMSE (`sqrt(mean(sqrt(dist)))` with per-cell affines). "
        "Depth-GSP reads C++ debug RMSE on its optimized meshes; DSFN recomputes the same formula on DSFN "
        "warp meshes (`I_mat` homography + TPS mesh) with SIFT matches on the aligned input pair. "
        "NIQE uses pyiqa on final panorama images.",
        "",
    ]
    (output_root / "method_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DSFN StitchBench General outputs with MDR/NIQE tables.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dsfn-root", default="outputs/stitchbench_general")
    parser.add_argument("--output-root", default="outputs/stitchbench_general")
    parser.add_argument("--depth-gsp-root", default=DEFAULT_DEPTH_GSP_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warp-ckpt", default=str(default_warp_ckpt()))
    parser.add_argument("--max-input-edge", type=int, default=2048)
    parser.add_argument("--skip-mdr", action="store_true")
    parser.add_argument("--skip-niqe", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = evaluate_dsfn(args)
    write_csv(output_root / "per_pair.csv", rows, list(PER_PAIR_COLUMNS))
    write_by_category(output_root, rows)
    comparison_rows = write_paper_comparison(output_root, rows)
    write_report(output_root, comparison_rows)
    if not args.skip_compare:
        compare_to_depth_gsp(output_root, rows, Path(args.depth_gsp_root))
    print(f"Wrote DSFN MDR/NIQE evaluation to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
