# MUTE - Mask-free Unified Token Editing

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
