"""Restore archived frozen source references and encode missing evaluation caches."""
import hashlib
import json
import shutil
from pathlib import Path
import run_frozen_evaluation as evaluation


def main():
    evaluation.check_freeze()
    archive=json.loads((evaluation.BASE/"source_archive.json").read_text())
    if evaluation.digest(evaluation.BASE/"masks.json") != archive["source_masks_manifest_sha256"]:
        raise ValueError("Source-mask manifest differs from archive record")
    for name,expected in archive["files_sha256"].items():
        if evaluation.digest(evaluation.BASE/name) != expected:
            raise ValueError(f"Archived source file changed: {name}")
    masks=json.loads((evaluation.BASE/"masks.json").read_text())
    for row in masks["cases"]:
        archived=evaluation.BASE/"source_archive"/f"{row['id']}_seed{row['seed']}"
        destination=(evaluation.ROOT/row["source"]).parent
        destination.mkdir(parents=True,exist_ok=True)
        for name in ["source.png","source_original.png","manifest.json"]:
            target=destination/name
            if target.exists():
                if evaluation.digest(target) != evaluation.digest(archived/name):
                    raise ValueError(f"Existing frozen source differs; no overwrite: {target}")
            else:
                shutil.copy2(archived/name,target)
    evaluation.masks_manifest()
    bank=evaluation.OUTPUT/"conditioning"
    if not (bank/"bank_manifest.json").exists():
        if bank.exists():
            raise ValueError("Existing incomplete conditioning directory; preserve it")
        if not evaluation.execute(["scripts/encode_development_bank.py","--protocol",str(evaluation.BASE/"protocol.json"),
                                   "--output",str(bank)],evaluation.OUTPUT,"replay_conditioning.log"):
            raise RuntimeError("Frozen conditioning replay failed; see log")
    if json.loads((bank/"bank_manifest.json").read_text())["status"] != "complete":
        raise ValueError("Conditioning bank incomplete")
    print("Frozen source archive and masks verified; conditioning ready; existing sources unchanged")


if __name__ == "__main__":main()
