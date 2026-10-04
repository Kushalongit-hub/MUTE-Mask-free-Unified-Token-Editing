# MUTE implementation checkpoint

Started 4 October 2026. Target research freeze: 27 October.

## Implemented first milestone

- Configurable generated-source / changed-target experiment using identical initial noise.
- Cross-attention capture delegates actual computation to the installed processor.
- Positive CFG branch, text padding, normalized probabilities, multi-subword phrase matching,
  and aggregation across explicitly selected layers and denoising calls.
- Ordinary, directional, symmetric, and reverse-direction self-attention controls.
- Saved source and edited images, localization, comparison sheet, versions, config,
  timings, hashes, memory measurements, and explicit failed-run manifests.
- Degenerate localization/output raises an error; no arbitrary patch fallback.
- Tests isolate directional information flow and verify tiny PixArt equivalence with
  intervention disabled. This is not evidence of pretrained editing quality.

## Run from this directory

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe -m mute.run --help
..\.venv\Scripts\python.exe -m mute.run --output runs\apple_seed42 --offload
```

The last command downloads the pretrained PixArt checkpoint if unavailable. On the
8 GB GPU use CPU offload. Model loading may need substantial disk space and RAM.
Use `--revision` with an immutable commit for frozen research runs, and
`--local-files-only` once weights are cached. Output directories must be new.

## Scope and interpretation

This is an initial controlled generated-image editing experiment. It changes a
prompt while restarting from the same seeded noise; it does not invert a supplied
real image or inject a preserved source trajectory. The source localization is
fixed during target denoising. Background MAE uses the inferred mask and is only
a diagnostic; it cannot establish edit success or independent preservation.
Lower MAE is not useful if the requested change fails. No novelty or quality gain
has been established. Disabling intervention delegates to the original processor;
the code intentionally supports standard PixArt AttnProcessor2_0, not every model.

## Next milestones

**Coverage checkpoint:** `RESEARCH_STATUS_2026-10-04_COVERAGE.md` adds explicit
mask margin/graded-ring ablations, with twelve passing tests. Margin 2 improves
mug recoloring and boundaries; defaults stay unchanged. Next freeze independent
evaluation masks and a broader held-out prompt/seed set, retaining restoration
and coverage as separate baselines for any directional claim.

**Restoration checkpoint:** `RESEARCH_STATUS_2026-10-04_RESTORATION.md` records
post-scheduler source-latent replacement and four-way ablations. Next address
object-mask coverage/boundary artifacts, freeze independent evaluation criteria,
and require a directional gain against restoration-only before retaining that
hypothesis as a contribution.

**Schedule checkpoint:** see `RESEARCH_STATUS_2026-10-04_SCHEDULE.md`. Late soft
intervention reduces severe artifacts; explicit apple and mug/cube prompts yield
visible semantic changes with ordinary and directional generation. Preservation
superiority is unproven. Next add source-trajectory restoration as a separate
ablation and compare the four combinations at successful edit quality.

**Updated pretrained checkpoint:** see `RESEARCH_STATUS_2026-10-04_PRETRAINED.md`.
Three seeds now pass exact pretrained capture equivalence, with completed images
and comparisons. Full-schedule hard masks damage preservation and reliable color
editing is not established. The next gate is successful semantic baseline edits
and a controlled study of less disruptive intervention schedules.

MUTE now has a fully isolated environment with Torch 2.14.0+cu130, Torchvision
0.29.0+cu130, and twelve passing tests. Install a fresh Windows GPU environment with
`python -m venv .venv`, then use `.venv\Scripts\python.exe -m pip install -r
MUTE\requirements_windows.lock.txt`. The historical initial environment note below
describes the earlier checkpoint and is superseded by the isolated setup.

Validation on 4 October: six unit/integration tests passed on CPU; a CUDA fp16
directional-attention smoke check passed on the RTX 5060 Laptop GPU; pipeline
imports, CLI help, and Git whitespace checks passed. Environment: Python 3.14,
Torch `2.12.0.dev20260408+cu128`, Diffusers 0.40.0, Transformers 5.18.0,
Accelerate 1.15.0. The project `.venv` inherits the host Torch installation;
it was not a fully isolated or portable lockfile environment. Pretrained generation
had not yet been executed at that earlier checkpoint.

1. Run full pretrained generation with capture disabled/enabled to confirm equivalent
   output; inspect finiteness and localization overlays on several seeds.
2. Execute color edits and inspect edit success jointly with background drift.
3. Add independently defined evaluation masks and edit-success scoring; separate
   development scenes from frozen test scenes.
4. If preservation is inadequate, implement source-feature/trajectory restoration as
   a separate ablation, not an unexplained part of the directional method.
5. Establish directional vs symmetric vs reverse controls at matched edit success.
6. Audit closest prior methods before naming a novelty contribution.
7. October 17: retain or reject the directional hypothesis based on repeatable results.
   October 18–22: final evaluation. October 23–27: report and reproducibility freeze.

Preserve legacy scripts as historical experiments. Do not use their black saved
dynamic image as a successful result. The separate 3D project is owned by Kilo;
coordinate long GPU batches and defer integration until standalone evidence exists.
