# Reproduce DSFN on StitchBench General (MDR/NIQE)

This repository includes a StitchBench General runner for the pretrained DSFN Warp + Fusion pipeline. Generated panoramas and evaluation tables stay under `outputs/`, which is ignored by git.

Evaluation follows the Depth-GSP / OBJ-GSP reference format for **MDR and NIQE only**. PSNR, SSIM, LPIPS, and SIQE are not computed.

## Environment

Use a CUDA-enabled Python environment. On this machine, the working venv is:

```powershell
C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe
```

Install dependencies:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-reproduce.txt
```

Pretrained weights should already be present at:

- `Warp/model/epoch001_model_warp.pth`
- `Fusion/model/epoch002_model_fusion.pth`

## Step 1: Inference (100 manifest pairs)

The runner reads the OBJ-GSP manifest (100 General pairs) and writes only `panorama.png` per dataset:

```powershell
python -m dsfn_py.run_stitchbench_general `
  --manifest C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035\manifest.csv `
  --out outputs\stitchbench_general `
  --device cuda `
  --max-input-edge 2048
```

On GPUs with about 11 GB memory, keep the default `--max-input-edge 2048` to avoid OOM on very large StitchBench inputs.

Output layout:

```
outputs/stitchbench_general/
  <dataset>/
    panorama.png
  failed_runs.csv
```

## Step 2: MDR/NIQE evaluation

```powershell
python -m dsfn_py.evaluate_stitchbench_mdr_niqe `
  --manifest C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035\manifest.csv `
  --dsfn-root outputs\stitchbench_general `
  --output-root outputs\stitchbench_general `
  --depth-gsp-root C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035 `
  --device cuda
```

This writes Depth-GSP-compatible tables into the same output root:

- `per_pair.csv`
- `by_category.csv`
- `paper_comparison.csv`
- `report.md`
- `method_pair_comparison.csv`
- `method_category_comparison.csv`
- `method_comparison.md`

### Metric definitions

| Metric | DSFN evaluation |
|--------|-----------------|
| **NIQE** | `pyiqa.create_metric("niqe")` on `panorama.png` |
| **MDR** | OBJ-GSP mesh RMSE (`MultiImage::getRMSE`) recomputed on DSFN warp meshes with SIFT correspondences |
| **warping_residual_avg/sd** | Mesh line-fit warping residual (`MultiImage::getWarpingResidual`) on DSFN canvas meshes |

MDR uses the same formula as Depth-GSP/OBJ-GSP C++ debug `{dataset}-RMSE-[DPS].txt`. Feature correspondences come from SIFT rather than the C++ matcher, so per-pair values may differ slightly from the baseline table.

## Smoke Test

```powershell
python -m dsfn_py.run_stitchbench_general `
  --manifest C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035\manifest.csv `
  --out outputs\stitchbench_general_smoke `
  --dataset SVA-01_chess `
  --device cuda

python -m dsfn_py.evaluate_stitchbench_mdr_niqe `
  --manifest C:\Users\22499\Documents\GitHub\OBJ-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035\manifest.csv `
  --dsfn-root outputs\stitchbench_general_smoke `
  --output-root outputs\stitchbench_general_smoke `
  --device cuda
```

Check `outputs/stitchbench_general_smoke/SVA-01_chess/panorama.png` and one row in `per_pair.csv`.

## Notes

- Inference uses the official DSFN Warp/Fusion code paths and does not require depth maps for stitching.
- MDR evaluation reruns the DSFN warp stage and applies the OBJ-GSP mesh RMSE formula; it does not read depth maps.
- The manifest covers **100 pairs** across 12 categories; the General folder has 110 scene directories, but NISwGSP-* and other extras are excluded from this benchmark split.
- This is a zero-shot evaluation using UDIS-D pretrained weights; it does not reproduce paper Table 1 numbers on UDIS-D/IVSD.
