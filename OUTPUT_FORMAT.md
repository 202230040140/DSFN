# Compatible benchmark outputs

This project exposes `reproduce_outputs.py` to publish its native outputs using the current benchmark image contract. Run native inference first, then export and evaluate. The collector does not run inference and never supplies another algorithm output for a missing task.

```powershell
& 'C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe' reproduce_outputs.py --source-root NATIVE_OUTPUT_ROOT --result-root D:\StitchBench_Result
# Optional: --scene SCENE; --manifest MANIFEST; --pair PAIR_ID for GT data.
```

StitchBench and Parallax: `<result-root>/<scene>/dsfn/stitch_result.png`. HD3D and LPS: `<result-root>/<scene>/pair_<id>/<method-folder>/stitch_result.png`; after evaluation also `aligned_to_gt.png` and `valid_mask.png`. Existing GT folders preserve `UDIS` and `DeepRectangling` capitalization. Output files are decoded, encoded as genuine RGB PNGs, verified, and atomically published. Missing/corrupt native images clear the old final panorama and old alignment. A fresh export clears local old scores and writes an explicit metrics-stale marker in `_work/output_exports`; do not publish root tables until evaluation has completed.

For the seven repaired methods, use `MethodManagement/tools/rebuild_repaired_benchmarks.py` instead. It performs fresh inference, checks run/input/code/checkpoint provenance on resume, calculates the existing metric protocols and replaces only selected rows. Do not use this standalone collector to republish a completed repaired run.

All other existing algorithm results and scores remain unchanged during the seven-method rerun.

The exporter uses the shared Python runtime above. Native inference retains its own project environment. See the per-project audit in MethodManagement/docs/datasets/ALGORITHM_OUTPUT_COMPATIBILITY.md.
