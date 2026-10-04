"""Export all frozen-evaluation proxy rows and unmodified-image contact sheets."""
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/"reports/frozen_evaluation_v1.json").read_text())
    protocol=json.loads((ROOT/"evaluation/frozen_v1/protocol.json").read_text())
    columns=["case","seed","variant","status","source_valid","target_color_coverage","colored_region_iou",
             "colored_region_centroid_shift_px","colored_region_area_ratio",
             "independent_color_mask_background_rgb_mae","joint_proxy_pass","error"]
    with (ROOT/"reports/frozen_evaluation_v1.csv").open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(result["rows"])
    conditions=[("Source","source","m2"),("Ordinary","none","m2"),("Directional only","directional","m2"),
                ("Restoration: original mask","none_restored","m0"),
                ("Restoration: 32px margin","none_restored","m2"),
                ("Margin + directional","directional_restored","m2")]
    for seed in protocol["seeds"]:
        canvas=Image.new("RGB",(256*len(conditions),300*len(protocol["cases"])),"white")
        draw=ImageDraw.Draw(canvas)
        for row,case in enumerate(protocol["cases"]):
            draw.text((8,300*row+5),f"{case['id']}, seed {seed}, red -> {case['target_color']}",fill="black")
            for col,(label,filename,group) in enumerate(conditions):
                draw.text((256*col+8,300*row+25),label,fill="black")
                path=ROOT/"runs/evaluation_v1"/f"{case['id']}_seed{seed}_{group}"/f"{filename}.png"
                if path.exists():
                    with Image.open(path) as image:
                        canvas.paste(image.convert("RGB").resize((256,256),Image.Resampling.LANCZOS),(256*col,300*row+44))
                else:
                    draw.text((256*col+8,300*row+80),"MISSING/FAILED",fill="red")
        canvas.save(ROOT/f"reports/frozen_evaluation_v1_seed{seed}.png")
    headers=["Method","Cases","Proxy passes","Median color coverage","Median color-region IoU","Median background MAE"]
    labels={"ordinary":"Ordinary generation", "directional":"Directional attention only",
            "restoration_m0":"Restoration: original mask", "restoration_m2":"Restoration: 32px margin",
            "combined_m2":"32px restoration + directional"}
    lines=["# Frozen evaluation v1 results","",
           "Restoration improves background preservation, but reliable whole-object recoloring and geometry retention are not established. Directional attention shows no clear added benefit in this eight-case screen.","",
           "Eight unseen prompt/seed cases, five variants, with frozen source-color masks independent of attention localization.",
           "These are automated color-surface and geometry proxies, not independently rated semantic success or full-object annotations.","",
           "| "+" | ".join(headers)+" |","|---|---:|---:|---:|---:|---:|"]
    for variant,s in result["summary"].items():
        metrics=[s["median_target_color_coverage"],s["median_colored_region_iou"],s["median_independent_color_mask_background_rgb_mae"]]
        values=[f"{v:.4f}" if v is not None else "—" for v in metrics]
        lines.append(f"| {labels[variant]} | {s['all_cases']} | {s['joint_proxy_passes']} | "+" | ".join(values)+" |")
    lines += ["","A proxy pass requires color coverage >=0.8, colored-region IoU >=0.7, centroid shift <=16 pixels, and background MAE <=0.03 simultaneously.",
              "Invalid source masks and failed/missing runs do not pass and remain in all-case denominators. Medians use available valid-source rows; undefined centroids after failed recoloring are not imputed.",
              "Background uses each frozen source-color mask dilated by 8 pixels, identical across methods. It is independent of the intervention mask but is not an independently annotated whole-object/background partition.",
              "Target hue components may include colored background spillover. Thin, dark and white object parts may be missed. Geometry and boundary conclusions require source silhouette annotations and blinded target ratings.",
              "This small screen supports descriptive comparisons only; no significance or novelty claim.","",
              "The two margin-2 proxy passes are teapot seed 4099 and backpack seed 4099. Inspection finds a broken/partly red handle on the teapot and changed pockets/straps/design on the backpack. Proxy passes must not be presented as validated clean same-object edits.",
              "[Implementing assistant visual notes](../evaluation/frozen_v1/VISUAL_OBSERVATIONS.md) are explicitly not independent ratings. The reviewer packet contains blank ratings and 40 paired images with method names omitted; independent review remains pending.","",
              "[Frozen protocol](../evaluation/frozen_v1/protocol.json), [hash freeze](../evaluation/frozen_v1/freeze.json), [source mask freeze](../evaluation/frozen_v1/masks.json)",
              "[Every case and metric](frozen_evaluation_v1.csv), [full machine record](frozen_evaluation_v1.json)",""]
    for seed in protocol["seeds"]:
        lines.append(f"![All methods, seed {seed}](frozen_evaluation_v1_seed{seed}.png)")
    (ROOT/"reports/FROZEN_EVALUATION_V1.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
