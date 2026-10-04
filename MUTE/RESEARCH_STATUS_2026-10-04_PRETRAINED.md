# Pretrained PixArt development check — 4 October 2026

## Outcome

Pretrained inference now runs locally and attention capture leaves source generation
unchanged. All three tested seeds passed exact decoded-pixel equivalence between
original attention processors and capture wrappers. The current hard, full-schedule
directional mask does not demonstrate a preservation benefit: it changes object
placement and background substantially. Symmetric and reverse isolation introduce
severe visible artifacts. Reliable red-to-green editing is not established even
for the ordinary target-prompt baseline.

This completes the pretrained execution/capture sanity milestone. It does not
complete the successful-edit or research-validation milestones.

## Protocol and artifacts

See `PRETRAINED_PROTOCOL_2026-10-04.md`. Model revision:
`50f702106901db6d0f8b67eb88e814c56ded2692`; all four large weight files were verified
against official Hub LFS SHA-256 values. The resumable downloader stores provenance
in `checkpoints/pixart512/source.json`. Weight storage is fp16; CUDA computation is
bf16. The installed Diffusers loader reports one unused legacy checkpoint tensor,
`caption_projection.y_embedding`; this compatibility notice is retained rather
than claiming identity with the authors' original implementation.

Source: `a red apple on a wooden table`; target: `a green apple on a wooden table`.
Resolution 512x512, 30 steps, guidance 4.5, threshold 0.3. Cross-attention capture
averages 12 observations: layers 6/12/18/24 at source calls 5/10/15. Each target uses
the same seeded initial noise as its source. The inferred mask is fixed and applied
to every self-attention layer throughout target denoising. No preserved source
features, trajectory, real-image inversion, or 3D inputs are used.

Seed 42 was the initial development run. Seeds 17 and 123 were selected after
viewing seed 42, with unchanged settings. These are exploratory checks on one
prompt pair, not an independently sampled or held-out benchmark.

Completed run directories:

- `runs/apple_seed42_verified_v2/`
- `runs/apple_seed17_verified/`
- `runs/apple_seed123_verified/`

Each contains source and original-source images, four target images, a labeled
comparison sheet, attention heatmap, binary mask, localization overlay, raw patch
arrays, and a manifest with settings, version/code/model hashes, timing, and memory.
`reports/pretrained_development_2026-10-04.json` summarizes the manifests.

## Measurements

All three source comparisons have zero maximum and mean decoded-pixel difference.
Seed-42 source image SHA-256 matches exactly:
`609f6c477121b3d98bb8c1a25f2f00d8423a013df1c3e1dfb480fa5293d6043c`.

| Seed | Active patches / 1024 | Ordinary background MAE | Directional | Symmetric | Reverse |
|---|---:|---:|---:|---:|---:|
| 42 | 198 | 0.0385 | 0.1023 | 0.1017 | 0.1419 |
| 17 | 220 | 0.0622 | 0.1243 | 0.1231 | 0.1223 |
| 123 | 151 | 0.0306 | 0.0647 | 0.0651 | 0.0924 |

MAE measures absolute RGB change on a 0–1 scale outside the inferred source mask.
This is a diagnostic, not an independent preservation metric. Ordinary generation
has lower background drift here but generally fails to make the apple green;
therefore these numbers do not compare methods at matched edit success and cannot
establish overall editing superiority.

The 12 target generations took 5.96–9.01 seconds each, including denoising,
decoding, offload transfers, and image saving; initial model/prompt encoding is
excluded. Maximum observed image-stage PyTorch reserved memory was 1.541 GB
(decimal), allocated about 1.324 GB. These figures exclude CUDA context, other
processes, and the initial text-encoding stage. Warm/cold load conditions differ;
this is not a controlled latency benchmark or a total-device memory claim.

## Visual inspection

The source images contain plausible red apples on wooden tables. Source masks
roughly localize the apple, with coarse patch boundaries and some surrounding
pixels; there is no independently annotated mask IoU measurement.

- Ordinary target generation retains a plausible scene but leaves the apple mostly
  red; seed 123 has a small yellow/green area near the top.
- Directional masking reduces apple size and shifts its placement to the right
  across these examples. Seed 42 has partial green coloring, seed 17 has a large
  green leaf-like region, and seed 123 remains largely red. Background geometry
  and wood patterns change visibly.
- Symmetric and reverse modes show repeated/fractured object content, patch-aligned
  discontinuities, and conspicuous background artifacts.

These are assistant visual observations, not independent human annotations or
validated semantic edit scores. The result supports investigating the disruption
caused by hard isolation; it does not disprove every directional-attention variant.

## Engineering fixes and environment

The first download route stalled; verified range transfers completed the weights.
During setup the inherited system PyTorch changed to a CPU build. MUTE now uses a
fully separate `.venv` with `include-system-site-packages = false`, local Torch
2.14.0+cu130 and Torchvision 0.29.0+cu130. No 3D-project files were modified.
`requirements_cuda_windows.txt` specifies the GPU build and
`requirements_windows.lock.txt` records 47 installed package versions.

Seven unit/integration tests pass, including a small complete pipeline using
precomputed conditioning without a text encoder; `pip check` reports no broken
requirements. Real-model testing identified and fixed negative-prompt argument
handling and CPU attention-mask placement. The failed initial inference remains
in `runs/apple_seed42_verified/manifest.json`; its successfully encoded prompts
were reused for completed runs through `conditioning.safetensors`.

Large T5 text-encoder modules stream to CUDA for prompt encoding, then are released.
Saved conditioning avoids reloading T5 for subsequent seeds. Embedding reuse checks
prompt/model/revision/CFG/dtype identity. Text embeddings, denoising latents, decoded
arrays, and degenerate localization/output are checked explicitly.

## Next development gates

1. Establish development prompt pairs on which ordinary target conditioning
   consistently performs the requested semantic change. Inspect embedding differences
   and caption behavior as part of that check; do not tune a final test set.
2. Add same-prompt intervention controls to measure disruption from attention
   restriction alone.
3. Test interventions restricted to later timesteps and selected layers, then soft
   penalties versus hard negative-infinity masking. Keep ordinary, symmetric, and
   reverse controls and record each changed configuration.
4. If necessary, add source-feature/trajectory restoration as an independently
   ablated component. Do not attribute its benefit to directionality by default.
5. Introduce independent evaluation masks and edit-success scoring before claiming
   preservation improvements. Maintain separate development and final cohorts.

Do not integrate 3D or make a novelty claim on the basis of this negative pilot.
