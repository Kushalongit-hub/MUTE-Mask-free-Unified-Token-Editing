# Pretrained development check: 4 October 2026

This is an exploratory development experiment, not the frozen research test.
Model: PixArt-alpha/PixArt-XL-2-512x512, revision
`50f702106901db6d0f8b67eb88e814c56ded2692`. Downloaded weight files must pass
the official Hub LFS SHA-256 checks. fp16 storage, bf16 CUDA computation.

## Fixed initial experiment

- Source: `a red apple on a wooden table`.
- Target: `a green apple on a wooden table`.
- Localization phrase: `apple`, verified against the real tokenizer at index 3.
- Seed 42; 512x512; 30 denoising steps; guidance 4.5.
- Capture source calls 5, 10, 15 in transformer layers 6, 12, 18, 24.
- Mean normalized token attention, min/max normalization, threshold 0.3.
- Fixed source mask during target generation. No source-feature restoration.
- Modes: ordinary, directional, symmetric, reverse.

Before interpreting edits, generate the source once using original processors and
once using capture wrappers with identical noise. Require exactly equal saved
pixels. Check text embeddings and denoising latents for finite values; reject
degenerate output and masks. Save failures and all successful outputs.

Inspect source-image plausibility, whether localization covers the apple, whether
the target apple changes color, and whether non-target content changes. Background
RGB MAE under inferred masks is diagnostic only. It does not supply an independent
preservation score and must be considered together with edit success and quality.
Any settings changed after inspecting results become a new development run.

After inspecting seed 42, seeds 17 and 123 were selected for additional development
sanity checks with unchanged settings. These are post-hoc checks, not a predeclared
held-out cohort.

## Commands

From `MUTE/`:

```powershell
..\.venv\Scripts\python.exe scripts\download_pixart.py --output checkpoints\pixart512 --workers 16
..\.venv\Scripts\python.exe -m mute.run --model checkpoints\pixart512 --local-files-only --output runs\apple_seed42_verified --offload --verify-capture
```

The resumable downloader checks HTTP Content-Range, chunk length, assembled length,
and official LFS SHA-256. Chunk files remain available to resume interruption.
Its `source.json` records the immutable model revision and file hashes.

The runner streams text-encoder submodules onto CUDA, precomputes both prompts,
then unloads the text encoder and offloads the smaller denoising/decoding models.
Model initialization and prompt encoding are included in initial load time;
per-image timing excludes that stage. PyTorch memory measurements exclude other
processes and non-PyTorch GPU allocations.

## Delivery gate

This phase is complete only after pretrained capture equivalence passes and the
source/mask/edit images have been visually inspected. Tests on random tiny models
alone do not complete it. If the directional method damages output, report it and
continue development; do not present lower drift from failed edits as success.
