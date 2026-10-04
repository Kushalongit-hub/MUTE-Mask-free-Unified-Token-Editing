# MUTE - Mask-free Unified Token Editing

## New controlled implementation (4 October 2026)

Start with [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). The new `mute/`
package provides a configurable source-to-target prompt experiment, corrected
attention capture, and ordinary/directional/symmetric/reverse ablations. Run
`..\.venv\Scripts\python.exe -m mute.run --help` from this directory.
The scripts described below are legacy prototypes. Their results do not establish
editing quality, background preservation, or novelty.

Pretrained checks are now complete on three development seeds; all pass exact
capture equivalence. The current hard attention masks damage the images and do
not establish an editing benefit. See
[the measured checkpoint](RESEARCH_STATUS_2026-10-04_PRETRAINED.md) and
[the experiment protocol](PRETRAINED_PROTOCOL_2026-10-04.md).

The second development phase adds intervention intervals, finite attention
penalties, and unchanged-prompt controls. Explicit apple wording and red-to-blue
mug/cube prompts establish visible semantic baselines. Late soft intervention
reduces artifacts, but preservation superiority is still unproven. See
[the schedule results](RESEARCH_STATUS_2026-10-04_SCHEDULE.md).

Example (from this directory, after downloading weights):

```powershell
..\.venv\Scripts\python.exe -m mute.run --model checkpoints\pixart512 --local-files-only --offload --output runs\my_late_soft_run --intervention-start 15 --penalty 2 --identity-controls
```

`--intervention-start` and `--intervention-end` count zero-based denoising calls;
the end is exclusive. Finite penalties subtract from blocked attention logits.
Unchanged-prompt controls quantify disturbance caused by the intervention itself.

Source-trajectory restoration is now available through `--restore-background`.
It adds restoration-only and combined variants with separately saved source
latents. Initial results preserve the background much better, while incomplete
object coverage and boundary artifacts remain. See
[the restoration checkpoint](RESEARCH_STATUS_2026-10-04_RESTORATION.md) and
[its protocol](PROTOCOL_RESTORATION_DEVELOPMENT_2026-10-04.md).

Coverage development adds `--mask-margin 2` (a 32-pixel editable-core radius) and
optional `--mask-feather 1` for restoration. These fix visible mug handle remnants
in development examples, with increased background drift and geometry changes.
The original evaluation mask stays fixed. See
[the coverage checkpoint](RESEARCH_STATUS_2026-10-04_COVERAGE.md).

Research codebase for mask-free unified token editing in text-to-image diffusion models, built on **PixArt-XL-2-512x512**.

## What this repo contains

- `run_dynamic_final.py` — end-to-end pipeline: capture cross-attention for a target token, build a dynamic active-token mask, and re-run PixArt with a custom self-attention mask that preserves only the attended spatial regions.
- `run_dynamic_v2.py` — same concept with stronger NaN/Inf validation and top-k patch inspection.
- `run_dynamic_hook.py` — alternative cross-attention capture via `register_forward_hook` instead of a custom attention processor.
- `custom_attention.py` — minimal repro of the directional attention mask in vanilla PyTorch SDPA.
- `debug_*.py` / `run_soundproof_*.py` / `test_cuda*.py` — targeted CUDA, slicing, CPU fallback, PixArt/SD3/SD1.5 probing, and earlier iterations of the masking idea.
- `soundproof_dynamic.png`, `soundproof_output.png`, `test_cpu.png` — saved inspection images.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install torch>=2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate
```

## Run

```bash
python run_dynamic_final.py
```

This will:
1. Load `PixArt-alpha/PixArt-XL-2-512x512` in fp16 on CUDA.
2. Tokenize `a red apple on a wooden table`, locate the `apple` token index.
3. Run 1 denoising step with a cross-attention capture processor to read where the model attends to `apple`.
4. Threshold the averaged cross-attention map into a boolean active patch mask over the 1024 latent patches.
5. Replace self-attention processors with `SoundproofAttnProcessor`, applying an asymmetric mask: anchor rows cannot attend to active columns, while active rows can attend freely.
6. Run 5 denoising steps with the dynamic mask and save the image.

## Key idea

Instead of a hand-drawn mask, extract a token-specific spatial mask from cross-attention at inference time, then feed it back as a structured self-attention mask during generation. No retraining, no token-specific fine-tuning.

## Hardware

Tested on RTX 5060 (sm_120, ~8 GB VRAM) with CUDA 12.8.
