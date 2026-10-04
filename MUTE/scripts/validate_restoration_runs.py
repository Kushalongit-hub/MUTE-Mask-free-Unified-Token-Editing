"""Audit saved execution invariants, without assigning semantic success labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from safetensors import safe_open
import torch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mute.trajectory import edit_patch_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.runs:
        data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if data["status"] != "complete" or not data["capture_verification"]["pixels_equal"]:
            raise ValueError(f"Incomplete or capture mismatch: {directory}")
        for name, record in data["runs"].items():
            if hashlib.sha256((directory / f"{name}.png").read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Image hash mismatch: {directory}/{name}")
            error = record.get("final_protected_background_latent_max_abs", record.get("final_background_latent_max_abs"))
            if record.get("restoration_enabled") and error != 0:
                raise ValueError(f"Restored final background differs: {directory}/{name}")
        source_hash = data["runs"]["source"]["sha256"]
        if "restoration_mask" in data:
            with np.load(directory / "localization.npz") as localization:
                active = torch.from_numpy(localization["active"])
            with np.load(directory / "restoration_weights.npz") as weights:
                actual = torch.from_numpy(weights["weights"])
            expected = edit_patch_weights(active, data["config"].get("mask_margin", 0),
                                          data["config"].get("mask_feather", 0))
            if not torch.equal(actual, expected):
                raise ValueError(f"Saved mask weights do not match configuration: {directory}")
            if int((actual == 0).sum()) != data["restoration_mask"]["protected_patches"]:
                raise ValueError("Protected mask count mismatch")
        for name in ["source_original", "none_identity", "none_restored_identity"]:
            if data["runs"][name]["sha256"] != source_hash:
                raise ValueError(f"Source identity mismatch: {directory}/{name}")
        path = directory / "source_trajectory.safetensors"
        if hashlib.sha256(path.read_bytes()).hexdigest() != data["trajectory"]["sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {directory}")
        with safe_open(path, framework="pt", device="cpu") as saved:
            if len(saved.keys()) != data["config"]["steps"]:
                raise ValueError("Missing source steps")
            if json.loads(saved.metadata()["timesteps"]) != data["trajectory"]["timesteps"]:
                raise ValueError("Schedule metadata mismatch")
            for i in range(data["config"]["steps"]):
                latent = saved.get_tensor(str(i))
                if list(latent.shape) != data["trajectory"]["shape"] or not torch.isfinite(latent).all():
                    raise ValueError("Invalid saved source latent")
        print(f"PASS {directory.name}: capture, identities, image/trajectory hashes, finite complete trajectory, protected final background latents")


if __name__ == "__main__":
    main()
