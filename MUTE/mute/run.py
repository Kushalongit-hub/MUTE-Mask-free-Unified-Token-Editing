"""Run with python -m mute.run from MUTE/. This is generated-image editing."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from .attention import AttentionStore, CaptureProcessor, DirectionalProcessor
from .trajectory import SourceTrajectory, latent_active_mask, edit_patch_weights, latent_edit_weight


def token_span(tokenizer, prompt, phrase):
    # Match complete token sequences, including words split into subwords.
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    query = tokenizer(phrase, add_special_tokens=False).input_ids
    matches = [i for i in range(len(ids) - len(query) + 1) if ids[i:i + len(query)] == query]
    if len(matches) != 1 or not query:
        raise ValueError("Target phrase must tokenize to one unambiguous span in the source prompt")
    start = matches[0]
    if start + len(query) > 119:
        raise ValueError("Target phrase is truncated by the pipeline")
    return list(range(start, start + len(query)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--revision", help="Pin an immutable model commit for research runs")
    parser.add_argument("--variant", default="fp16", help="Checkpoint weight variant")
    parser.add_argument("--verify-capture", action="store_true", help="Compare unwrapped and captured source pixels")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--source", default="a red apple on a wooden table")
    parser.add_argument("--target", default="a green apple on a wooden table")
    parser.add_argument("--phrase", default="apple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--intervention-start", type=int, default=0)
    parser.add_argument("--intervention-end", type=int)
    parser.add_argument("--penalty", type=float, default=float("inf"), help="Positive logit penalty; inf gives hard masking")
    parser.add_argument("--intervention-layers", type=int, nargs="+", help="Restrict self-attention intervention to these layers")
    parser.add_argument("--identity-controls", action="store_true", help="Also apply interventions to the unchanged source prompt")
    parser.add_argument("--restore-background", action="store_true", help="Add separate source-latent restoration ablations for each attention mode")
    parser.add_argument("--restoration-start", type=int, default=0)
    parser.add_argument("--restoration-end", type=int)
    parser.add_argument("--mask-margin", type=int, default=0, help="Restoration edit-core dilation radius in transformer patches")
    parser.add_argument("--mask-feather", type=int, default=0, help="Additional graded edit ring width in transformer patches")
    parser.add_argument("--modes", nargs="+", choices=["none", "directional", "symmetric", "reverse"],
                        default=["none", "directional", "symmetric", "reverse"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--conditioning-cache", type=Path, help="Reuse this runner's saved prompt embeddings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("Source and target prompts must differ")
    if args.steps < 2 or any(s < 0 or s >= args.steps for s in args.capture_steps):
        parser.error("Capture steps must lie within the denoising schedule")
    if not 0 < args.threshold < 1 or not np.isfinite(args.guidance) or args.guidance < 1:
        parser.error("Threshold must be in (0,1), guidance must be >= 1")
    if args.offload and args.device != "cuda":
        parser.error("CPU offload requires CUDA")
    if args.conditioning_cache and not args.offload:
        parser.error("Conditioning cache currently requires --offload")
    if not 0 <= args.intervention_start < args.steps or (args.intervention_end is not None and
            not args.intervention_start < args.intervention_end <= args.steps):
        parser.error("Invalid intervention step interval")
    if args.penalty <= 0 or args.penalty != args.penalty:
        parser.error("Penalty must be positive")
    if args.mask_margin < 0 or args.mask_feather < 0:
        parser.error("Mask margin and feather must be nonnegative")
    if (args.mask_margin or args.mask_feather) and not args.restore_background:
        parser.error("Mask coverage options require --restore-background")
    if not 0 <= args.restoration_start < args.steps or (args.restoration_end is not None and
            not args.restoration_start < args.restoration_end <= args.steps):
        parser.error("Invalid restoration interval")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) or (isinstance(v, float) and not np.isfinite(v)) else v
              for k, v in vars(args).items()}
    manifest = {"config": config, "status": "started", "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ["torch", "diffusers", "transformers", "accelerate"]},
                "input_type": "generated source; same initial noise; no real-image inversion",
                "metrics_warning": "Background changes use inferred localization, not independent evaluation masks",
                "runs": {}}
    manifest["code_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ["run.py", "attention.py", "trajectory.py"]}
    provenance_path = Path(args.model) / "source.json"
    if provenance_path.exists():
        manifest["model_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest_path = args.output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    try:
        from diffusers import PixArtAlphaPipeline
        from diffusers.models.attention_processor import AttnProcessor2_0
        dtype = getattr(torch, args.dtype) if args.device == "cuda" else torch.float32
        started = time.perf_counter()
        embeddings = {}
        if args.conditioning_cache:
            from safetensors import safe_open
            with safe_open(args.conditioning_cache, framework="pt", device="cpu") as cached:
                identity = {"source": args.source, "target": args.target,
                            "model": args.model, "revision": args.revision,
                            "cfg": args.guidance > 1, "dtype": args.dtype}
                if json.loads(cached.metadata()["identity"]) != identity:
                    raise ValueError("Conditioning cache does not match prompts/model/dtype/CFG")
                for index, prompt in enumerate([args.source, args.target]):
                    embeddings[prompt] = {key: cached.get_tensor(f"{index}.{key}")
                        if f"{index}.{key}" in cached.keys() else None
                        for key in ["prompt_embeds", "prompt_attention_mask", "negative_prompt_embeds",
                                    "negative_prompt_attention_mask"]}
        pipe = PixArtAlphaPipeline.from_pretrained(args.model, revision=args.revision,
                    local_files_only=args.local_files_only, dtype=dtype, variant=args.variant,
                    **({"text_encoder": None} if args.conditioning_cache else {}))
        if args.offload and not args.conditioning_cache:
            # T5-XXL cannot fit as a whole on the 8 GB GPU. Stream its submodules
            # for two prompt encodings, then release it before image generation.
            pipe.enable_sequential_cpu_offload()
            with torch.inference_mode():
                for prompt in dict.fromkeys([args.source, args.target]):
                    encoded = pipe.encode_prompt(prompt, do_classifier_free_guidance=args.guidance > 1,
                                                 device=torch.device("cuda"), clean_caption=False)
                    if any(t is not None and not torch.isfinite(t).all() for t in encoded):
                        raise ValueError("Text encoder produced nonfinite embeddings")
                    embeddings[prompt] = {key: value.detach().cpu() if value is not None else None
                        for key, value in zip(["prompt_embeds", "prompt_attention_mask",
                                              "negative_prompt_embeds", "negative_prompt_attention_mask"], encoded)}
            del encoded
            pipe.register_modules(text_encoder=None)
            # Drop T5 before removing the remaining hooks: restoring its meta
            # parameters first would unnecessarily reconstruct the full CPU model.
            pipe.remove_all_hooks()
            gc.collect()
            torch.cuda.empty_cache()
            from safetensors.torch import save_file
            save_file({f"{i}.{key}": value.contiguous() for i, prompt in enumerate([args.source, args.target])
                       for key, value in embeddings[prompt].items() if value is not None},
                      args.output / "conditioning.safetensors", metadata={"identity": json.dumps(
                          {"source": args.source, "target": args.target, "model": args.model,
                           "revision": args.revision, "cfg": args.guidance > 1, "dtype": args.dtype})})
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
        manifest["model_load_seconds"] = time.perf_counter() - started
        if args.device == "cuda":
            manifest["gpu"] = torch.cuda.get_device_name()
        originals = dict(pipe.transformer.attn_processors)
        available_layers = {int(key.split(".")[1]) for key in originals if key.endswith("attn1.processor")}
        if args.intervention_layers is not None and not set(args.intervention_layers) <= available_layers:
            raise ValueError("Requested intervention layers are not present in this model")
        if any(not isinstance(p, AttnProcessor2_0) for p in originals.values()):
            raise ValueError("This runner is validated for PixArt AttnProcessor2_0 only")
        indices = token_span(pipe.tokenizer, args.source, args.phrase)
        manifest["token_indices"] = indices
        store = AttentionStore(indices, args.guidance > 1, args.capture_steps)
        captures = dict(originals)
        trajectory = SourceTrajectory()
        if args.restore_background and pipe.scheduler.order != 1:
            raise ValueError("Trajectory callback requires scheduler.order == 1")
        for layer in args.layers:
            key = f"transformer_blocks.{layer}.attn2.processor"
            if key not in originals:
                raise ValueError(f"Missing cross-attention layer {layer}")
            captures[key] = CaptureProcessor(originals[key], store, layer)

        def generate(prompt, name, restore=False):
            for processor in pipe.transformer.attn_processors.values():
                if isinstance(processor, DirectionalProcessor):
                    processor.step = 0
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            callback_count = 0
            final_latents = None
            def check_latents(step, timestep, latents):
                nonlocal callback_count, final_latents
                if step != callback_count:
                    raise ValueError("Denoising callbacks must be sequential")
                callback_count += 1
                if not torch.isfinite(latents).all():
                    raise ValueError(f"{name}: nonfinite latents at denoising step {step}")
                if args.restore_background and name == "source":
                    trajectory.record(step, timestep, latents)
                if restore:
                    trajectory.restore(step, timestep, latents, edit_weights,
                                       args.restoration_start, args.restoration_end)
                if args.restore_background and step == args.steps - 1:
                    final_latents = latents.detach().cpu().clone()

            with torch.inference_mode():
                conditioning = ({**{key: value.to(args.device) if value is not None else None
                                    for key, value in embeddings[prompt].items()}, "negative_prompt": None}
                                if args.offload else {"prompt": prompt})
                array = pipe(**conditioning, generator=torch.Generator("cpu").manual_seed(args.seed),
                             num_inference_steps=args.steps, guidance_scale=args.guidance,
                             height=512, width=512, clean_caption=False,
                             callback=check_latents, callback_steps=1, output_type="np").images[0]
            if args.restore_background and callback_count != args.steps:
                raise ValueError("Trajectory does not cover every denoising step")
            if args.device == "cuda":
                torch.cuda.synchronize()
            if not np.isfinite(array).all():
                raise ValueError(f"{name}: nonfinite decoded image")
            image = Image.fromarray((array.clip(0, 1) * 255).round().astype("uint8"))
            if np.asarray(image).std() < 1:
                raise ValueError(f"{name}: degenerate image output")
            image.save(args.output / f"{name}.png")
            manifest["runs"][name] = {"seconds": time.perf_counter() - begin,
                "sha256": hashlib.sha256((args.output / f"{name}.png").read_bytes()).hexdigest()}
            if args.restore_background and name not in ["source", "source_original"]:
                reference = trajectory.latents[-1]
                background_latent = (~latent_active_mask(active, final_latents)).expand_as(final_latents)
                difference = (final_latents.float() - reference.float()).abs()[background_latent]
                manifest["runs"][name].update(restoration_enabled=restore,
                    final_background_latent_max_abs=float(difference.max()),
                    final_background_latent_mae=float(difference.mean()))
                protected = (latent_edit_weight(edit_weights, final_latents) == 0).expand_as(final_latents)
                manifest["runs"][name]["final_protected_background_latent_max_abs"] = float(
                    (final_latents.float() - reference.float()).abs()[protected].max())
            if args.device == "cuda":
                manifest["runs"][name].update(allocated_bytes=torch.cuda.max_memory_allocated(),
                                             reserved_bytes=torch.cuda.max_memory_reserved())
            save()
            return image

        if args.verify_capture:
            baseline_source = generate(args.source, "source_original")
        pipe.transformer.set_attn_processor(dict(captures))
        source = generate(args.source, "source")
        if args.restore_background:
            from safetensors.torch import save_file
            trajectory_path = args.output / "source_trajectory.safetensors"
            save_file({str(i): value for i, value in enumerate(trajectory.latents)}, trajectory_path,
                      metadata={"timesteps": json.dumps(trajectory.timesteps)})
            manifest["trajectory"] = {"steps": len(trajectory.latents),
                "timesteps": trajectory.timesteps, "shape": list(trajectory.latents[0].shape),
                "sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
                "method": "post-scheduler background latent replacement; target solver history retained"}
        if args.verify_capture:
            difference = np.abs(np.asarray(source).astype(int) - np.asarray(baseline_source).astype(int))
            manifest["capture_verification"] = {"pixels_equal": bool(not difference.any()),
                "max_pixel_difference": int(difference.max()), "mean_pixel_difference": float(difference.mean())}
            save()
            if difference.any():
                raise ValueError("Capture changed source output; investigate before comparing edits")
        heat, active = store.localize(args.threshold)
        edit_weights = edit_patch_weights(active, args.mask_margin, args.mask_feather)
        if not (edit_weights == 0).any():
            raise ValueError("Mask expansion leaves no protected background")
        side = int(active.numel() ** 0.5)
        if side * side != active.numel():
            raise ValueError("Expected a square PixArt patch grid")
        np.savez(args.output / "localization.npz", heat=heat.numpy(), active=active.numpy())
        np.savez(args.output / "restoration_weights.npz", weights=edit_weights.numpy())
        Image.fromarray((edit_weights.reshape(side, side).numpy() * 255).round().astype("uint8")).resize(
            source.size, Image.Resampling.NEAREST).save(args.output / "restoration_weights.png")
        Image.fromarray((heat.reshape(side, side).numpy() * 255).astype("uint8")).resize(
            source.size, Image.Resampling.NEAREST).save(args.output / "attention.png")
        mask = Image.fromarray((active.reshape(side, side).numpy() * 255).astype("uint8")).resize(
            source.size, Image.Resampling.NEAREST)
        mask.save(args.output / "mask.png")
        overlay = source.convert("RGBA")
        tint = Image.new("RGBA", source.size, (255, 0, 0, 0))
        tint.putalpha(mask.point(lambda value: 70 if value else 0))
        Image.alpha_composite(overlay, tint).convert("RGB").save(args.output / "localization_overlay.png")
        background = np.asarray(mask) == 0
        manifest["localization"] = {"active_patches": int(active.sum()), "patches": active.numel(),
                                    "captures": store.records}
        manifest["restoration_mask"] = {"full_edit_patches": int((edit_weights == 1).sum()),
            "graded_ring_patches": int(((edit_weights > 0) & (edit_weights < 1)).sum()),
            "protected_patches": int((edit_weights == 0).sum()),
            "evaluation_mask": "Original undilated source localization, fixed across coverage settings",
            "interpolation": "nearest; Chebyshev patch dilation and graded outer rings"}
        images = [source]
        image_labels = ["source"]
        for mode in args.modes:
            processors = dict(originals)
            for key in processors:
                if key.endswith("attn1.processor") and (args.intervention_layers is None or
                        int(key.split(".")[1]) in args.intervention_layers):
                    processors[key] = DirectionalProcessor(originals[key], active, mode,
                        args.intervention_start, args.intervention_end, args.penalty)
            pipe.transformer.set_attn_processor(dict(processors))
            for restore in ([False, True] if args.restore_background else [False]):
                variant = mode + ("_restored" if restore else "")
                edited = generate(args.target, variant, restore)
                delta = np.abs(np.asarray(edited).astype(float) - np.asarray(source).astype(float)) / 255
                manifest["runs"][variant]["background_rgb_mae_diagnostic"] = float(delta[background].mean())
                images.append(edited)
                image_labels.append(variant)
                if args.identity_controls:
                    name = variant + "_identity"
                    identity = generate(args.source, name, restore)
                    delta_identity = np.abs(np.asarray(identity).astype(float) - np.asarray(source).astype(float)) / 255
                    manifest["runs"][name]["background_rgb_mae_diagnostic"] = float(delta_identity[background].mean())
                    manifest["runs"][name]["whole_image_rgb_mae_diagnostic"] = float(delta_identity.mean())
                    images.append(identity)
                    image_labels.append(name)
                    if mode == "none" and not delta_identity.any():
                        manifest["runs"][name]["identity_pixels_equal"] = True
                    elif mode == "none":
                        raise ValueError("Ordinary/restoration-only unchanged prompt must reproduce source exactly")
        sheet = Image.new("RGB", (512 * len(images), 544), "white")
        draw = ImageDraw.Draw(sheet)
        for i, image in enumerate(images):
            sheet.paste(image, (512 * i, 32))
            label = image_labels[i]
            draw.text((512 * i + 12, 10), label, fill="black")
        sheet.save(args.output / "comparison.png")
        manifest["comparison_order"] = image_labels
        manifest["status"] = "complete"
        save()
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
