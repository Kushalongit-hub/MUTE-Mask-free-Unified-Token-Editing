"""Encode a fixed development prompt bank once with the pinned T5 encoder."""
import argparse
import json
from pathlib import Path
import time

import torch
from diffusers import PixArtAlphaPipeline
from safetensors.torch import save_file


PAIRS = {
    "apple": ("a red apple on a wooden table", "a green apple on a wooden table", "apple"),
    "apple_explicit": ("a red apple on a wooden table",
                       "a bright green apple with completely green skin on a wooden table", "apple"),
    "mug": ("a red ceramic mug on a wooden table", "a blue ceramic mug on a wooden table", "mug"),
    "cube": ("a red cube on a wooden table", "a blue cube on a wooden table", "cube"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="checkpoints\\pixart512")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, help="Use a frozen evaluation protocol's cases instead of development prompts")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = PAIRS
    if args.protocol:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        pairs = {case["id"]: (case["source"], case["target"], case["phrase"]) for case in protocol["cases"]}
    record = {"status": "encoding", "pairs": pairs, "model": args.model,
              "scope": "Frozen evaluation conditioning" if args.protocol else "Development prompt selection, not held-out evaluation"}
    manifest = args.output / "bank_manifest.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    begin = time.perf_counter()
    try:
        pipe = PixArtAlphaPipeline.from_pretrained(args.model, transformer=None, vae=None,
                dtype=torch.bfloat16, variant="fp16", local_files_only=True)
        pipe.enable_sequential_cpu_offload()
        prompts = list(dict.fromkeys([""] + [prompt for pair in pairs.values() for prompt in pair[:2]]))
        encodings = {}
        with torch.inference_mode():
            for prompt in prompts:
                result = pipe.encode_prompt(prompt, do_classifier_free_guidance=False,
                                            device=torch.device("cuda"), clean_caption=False)
                if not torch.isfinite(result[0]).all():
                    raise ValueError(f"Nonfinite encoding: {prompt}")
                encodings[prompt] = (result[0].cpu(), result[1].cpu())
                print("encoded", repr(prompt), flush=True)
        for name, (source, target, phrase) in pairs.items():
            tensors = {}
            for i, prompt in enumerate([source, target]):
                tensors[f"{i}.prompt_embeds"] = encodings[prompt][0].contiguous()
                tensors[f"{i}.prompt_attention_mask"] = encodings[prompt][1].contiguous()
                tensors[f"{i}.negative_prompt_embeds"] = encodings[""][0].clone().contiguous()
                tensors[f"{i}.negative_prompt_attention_mask"] = encodings[""][1].clone().contiguous()
            identity = {"source": source, "target": target, "model": args.model,
                        "revision": None, "cfg": True, "dtype": "bfloat16"}
            save_file(tensors, args.output / f"{name}.safetensors",
                      metadata={"identity": json.dumps(identity)})
        record.update(status="complete", seconds=time.perf_counter() - begin)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
