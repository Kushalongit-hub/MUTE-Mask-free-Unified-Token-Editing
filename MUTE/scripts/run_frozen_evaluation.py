"""Freeze, generate source-only masks, run, and score evaluation v1 sequentially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mute.evaluation import source_mask, score

BASE = ROOT / "evaluation/frozen_v1"
OUTPUT = ROOT / "runs/evaluation_v1"
FILES = ["evaluation/frozen_v1/protocol.json", "mute/run.py", "mute/attention.py",
         "mute/trajectory.py", "mute/evaluation.py", "scripts/run_frozen_evaluation.py",
         "scripts/encode_development_bank.py", "requirements_windows.lock.txt"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def check_freeze():
    frozen = json.loads((BASE/"freeze.json").read_text())
    for name, value in frozen["sha256"].items():
        if digest(ROOT/name) != value:
            raise ValueError(f"Frozen input changed: {name}; use a new evaluation version")
    return frozen


def execute(arguments, directory, log_name):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/log_name).open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable]+arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"FAILED ({result.returncode}): {directory/log_name}", flush=True)
    return result.returncode == 0


def command(case, seed, destination, margin, source_only=False):
    bank = OUTPUT/"conditioning"/f"{case['id']}.safetensors"
    args = ["-m", "mute.run", "--model", "checkpoints\\pixart512", "--local-files-only", "--offload",
            "--source", case["source"], "--target", case["target"], "--phrase", case["phrase"],
            "--seed", str(seed), "--conditioning-cache", str(bank), "--output", str(destination),
            "--restore-background", "--mask-margin", str(margin), "--identity-controls", "--verify-capture",
            "--intervention-start", "15", "--penalty", "2", "--modes", "none"]
    if margin == 2 and not source_only:
        args.append("directional")
    if source_only:
        args.append("--source-only")
    return args


def completed(path, phase=None):
    if not (path/"manifest.json").exists():
        return False
    data = json.loads((path/"manifest.json").read_text())
    return data["status"] == "complete" and (phase is None or data.get("phase") == phase)


def masks_manifest():
    data = json.loads((BASE/"masks.json").read_text())
    for row in data["cases"]:
        if "mask" in row and digest(ROOT/row["mask"]) != row["mask_sha256"]:
            raise ValueError("Frozen source evaluation mask changed")
        if "source" in row and digest(ROOT/row["source"]) != row["source_sha256"]:
            raise ValueError("Frozen source image changed")
    return data


def evaluate(protocol):
    masks = masks_manifest()
    lookup = {(r["id"],r["seed"]):r for r in masks["cases"]}
    rows = []
    variants = {"ordinary": ("m2","none"), "directional": ("m2","directional"),
                "restoration_m0": ("m0","none_restored"), "restoration_m2": ("m2","none_restored"),
                "combined_m2": ("m2","directional_restored")}
    for case in protocol["cases"]:
        for seed in protocol["seeds"]:
            item = lookup[(case["id"],seed)]
            for variant, (group,image_name) in variants.items():
                directory = OUTPUT/f"{case['id']}_seed{seed}_{group}"
                row = {"case":case["id"],"seed":seed,"variant":variant,"directory":str(directory.relative_to(ROOT))}
                try:
                    if not completed(directory) or "mask" not in item:
                        raise ValueError("Missing/failed target run or source mask")
                    manifest = json.loads((directory/"manifest.json").read_text())
                    if manifest["runs"]["source"]["sha256"] != item["source_sha256"]:
                        raise ValueError("Target-run source does not match frozen source")
                    with Image.open(ROOT/item["source"]) as source, Image.open(directory/f"{image_name}.png") as target:
                        mask = np.asarray(Image.open(ROOT/item["mask"]).convert("L")) > 0
                        row.update(score(source.convert("RGB"), target.convert("RGB"), mask,case["target_color"],protocol))
                    row.update(status="scored",image_sha256=digest(directory/f"{image_name}.png"))
                except Exception as error:
                    row.update(status="failed", error=str(error), joint_proxy_pass=False)
                rows.append(row)
    result = {"protocol":protocol["id"], "freeze_sha256":digest(BASE/"freeze.json"),
              "masks_manifest_sha256":digest(BASE/"masks.json"),
              "warning":"Automated colored-region proxies, not full-object geometry or independently rated semantic success", "rows":rows}
    summary = {}
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        valid = [r for r in group if r.get("source_valid")]
        summary[variant] = {"all_cases":len(group),"valid_source_cases":len(valid),
            "failed_runs":sum(r["status"] == "failed" for r in group),
            "joint_proxy_passes":sum(r["joint_proxy_pass"] for r in group)}
        for metric in ["target_color_coverage","colored_region_iou","colored_region_centroid_shift_px",
                       "colored_region_area_ratio","independent_color_mask_background_rgb_mae"]:
            values=[r[metric] for r in valid if r.get(metric) is not None]
            summary[variant]["median_"+metric]=float(np.median(values)) if values else None
    result["summary"]=summary
    write(ROOT/"reports/frozen_evaluation_v1.json",result)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["freeze","sources","targets","score"])
    args=parser.parse_args()
    protocol=json.loads((BASE/"protocol.json").read_text())
    if args.stage == "freeze":
        if (BASE/"freeze.json").exists():
            check_freeze();print("Existing freeze verified");return
        write(BASE/"freeze.json", {"created_utc":datetime.now(timezone.utc).isoformat(),
              "sha256":{name:digest(ROOT/name) for name in FILES}})
        print("Protocol, implementation, evaluator and environment hashes frozen");return
    check_freeze()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.stage == "sources":
        if (BASE/"masks.json").exists():
            masks_manifest();print("Existing source masks verified; no replacements");return
        bank=OUTPUT/"conditioning"
        if not (bank/"bank_manifest.json").exists():
            print("Encoding frozen prompt bank",flush=True)
            if not execute(["scripts/encode_development_bank.py","--protocol",str(BASE/"protocol.json"),
                            "--output",str(bank)],OUTPUT,"conditioning.log"):
                raise RuntimeError("Conditioning bank failed; preserve failure log")
        if json.loads((bank/"bank_manifest.json").read_text())["status"] != "complete":
            raise RuntimeError("Incomplete conditioning bank")
        records=[]
        sheet=Image.new("RGB",(256*len(protocol["seeds"]),284*len(protocol["cases"])),"white")
        draw=ImageDraw.Draw(sheet)
        for case_index,case in enumerate(protocol["cases"]):
            for seed_index,seed in enumerate(protocol["seeds"]):
                name=f"{case['id']}_seed{seed}"
                destination=OUTPUT/(name+"_source")
                print("Source only:",name,flush=True)
                success=completed(destination,"source_only")
                if not success and not destination.exists():
                    success=execute(command(case,seed,destination,0,True),OUTPUT,name+"_source.log")
                row={"id":case["id"],"seed":seed,"source_generation_complete":success}
                if success:
                    source_path=destination/"source.png"
                    with Image.open(source_path) as source:
                        source=source.convert("RGB")
                        mask=source_mask(source,protocol)
                        path=BASE/"masks"/(name+".png")
                        path.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(mask.astype("uint8")*255).save(path)
                        overlay=np.asarray(source).copy()
                        overlay[mask]=(overlay[mask].astype(float)*.6+np.array([0,255,0])*.4).astype("uint8")
                        sheet.paste(Image.fromarray(overlay).resize((256,256)),(256*seed_index,284*case_index+28))
                        draw.text((256*seed_index+8,284*case_index+8),name,fill="black")
                        validity=protocol["source_validity"]
                        row.update(source=str(source_path.relative_to(ROOT)),source_sha256=digest(source_path),
                            mask=str(path.relative_to(ROOT)),mask_sha256=digest(path),pixels=int(mask.sum()),
                            source_valid=validity["minimum_pixels"] <= int(mask.sum()) <= validity["maximum_area_fraction"]*mask.size)
                records.append(row)
        write(BASE/"masks.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
              "freeze_sha256":digest(BASE/"freeze.json"),"cases":records})
        sheet.save(OUTPUT/"source_mask_review.png")
        print("Source-only masks frozen; review overlays before targets",flush=True)
    elif args.stage == "targets":
        masks_manifest()
        for case in protocol["cases"]:
            for seed in protocol["seeds"]:
                for margin in [0,2]:
                    name=f"{case['id']}_seed{seed}_m{margin}"
                    destination=OUTPUT/name
                    if completed(destination):
                        print("Completed run retained:",name,flush=True);continue
                    if destination.exists():
                        print("Existing incomplete run retained:",name,flush=True);continue
                    print("Target variants:",name,flush=True)
                    execute(command(case,seed,destination,margin),OUTPUT,name+".log")
        evaluate(protocol)
    else:
        evaluate(protocol)


if __name__ == "__main__":
    main()
