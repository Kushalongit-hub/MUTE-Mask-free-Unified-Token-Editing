# Frozen evaluation v1

Eight source cases: four new object/prompt pairs at seeds 2027 and 4099. None of
these exact prompts/seeds was used in development. The selected restoration
margin and directional parameters come from the previously inspected examples.
This is a small frozen screen, not evidence of generalization across a benchmark.

Freeze protocol, runner and metric code hashes before generating sources. Generate
and inspect sources only, then derive/hash their evaluation masks before any target
generation. Keep all cases, including source failures or poor color masks. Do not
tune prompts, HSV thresholds, methods or pass thresholds using edited outputs.
If a method or protocol changes, use a new version and separate output directories.

Compare ordinary, directional only, zero-margin restoration, selected margin-2
restoration, and margin-2 restoration plus directional attention. The same noise,
conditioning, scheduler and source mask apply within each prompt/seed. Include
unchanged-prompt controls and exact capture checks. Repeated ordinary/directional
outputs across margins must have equal hashes. All run failures remain visible.

Automated evaluation masks use source red-color segmentation, independent of the
attention mask. This is a color-surface proxy, not a hand-labeled object mask.
Source shadows, reflections and thin/dark/white parts can make it inaccurate.
Review source overlays and retain those limitations; never repair a source mask
after seeing edited outputs. Target hue masks can confuse colored background
spillover with the object. IoU/centroid/area measure colored-region geometry only.
They cannot establish whole-object identity or shape, boundary quality, or clean
semantic edit success. Human/source-only silhouette annotations and independent
blinded target ratings remain necessary for those claims.

Report color coverage, colored-region IoU/centroid/area ratio and background MAE
together. Proxy threshold passes are labeled as such and are not benchmark success
labels. Preserve per-case rows and visual comparisons, not only averages. Count
failures in all-case denominators. Eight cases do not justify significance claims.

## Reproduce the frozen screen

The curated `source_archive/` keeps all eight source PNGs, their unwrapped capture
controls and source-run manifests. They are byte copies of the source images
hashed before targets, not newly generated references. Large checkpoints and
raw run outputs remain outside version control. A fresh checkout therefore needs
the pinned local checkpoint and the isolated locked environment, then from MUTE/:

```powershell
..\.venv\Scripts\python.exe scripts\prepare_frozen_replay.py
..\.venv\Scripts\python.exe -u scripts\run_frozen_evaluation.py targets
..\.venv\Scripts\python.exe scripts\build_frozen_report.py
..\.venv\Scripts\python.exe scripts\build_frozen_review.py
```

The preparation wrapper verifies existing frozen sources or restores missing
references from the archive without overwriting differing files. It encodes
missing conditioning with the frozen encoder script. The target runner retains
complete/failed existing runs, checks frozen hashes, and rejects changed source
images. Exact-image reproduction is checked against the recorded environment;
hardware/runtime changes may prevent byte equality and must not be silently
accepted as the original experiment. Existing review packets are not overwritten.

Measured results: [report](../../reports/FROZEN_EVALUATION_V1.md).
Independent review sheet: `runs/evaluation_v1/blinded_review/ratings.csv`.
It remains blank; automated proxy passes are not independent human ratings.
