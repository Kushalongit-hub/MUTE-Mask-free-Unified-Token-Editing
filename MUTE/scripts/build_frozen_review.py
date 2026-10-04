"""Create unlabeled-method source/target pairs and a blank independent review sheet."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/"reports/frozen_evaluation_v1.json").read_text())
    protocol=json.loads((ROOT/"evaluation/frozen_v1/protocol.json").read_text())
    cases={case["id"]:case for case in protocol["cases"]}
    directory=ROOT/"runs/evaluation_v1/blinded_review"
    directory.mkdir(exist_ok=False)
    ordered=sorted(data["rows"],key=lambda row:hashlib.sha256(
        f"{data['freeze_sha256']}:{row['case']}:{row['seed']}:{row['variant']}".encode()).hexdigest())
    names={"ordinary":"none","directional":"directional","restoration_m0":"none_restored",
           "restoration_m2":"none_restored","combined_m2":"directional_restored"}
    ratings=[];mapping=[]
    for index,row in enumerate(ordered):
        identifier=f"E{index+1:03}"
        source=ROOT/"runs/evaluation_v1"/f"{row['case']}_seed{row['seed']}_source/source.png"
        target=ROOT/row["directory"]/f"{names[row['variant']]}.png"
        canvas=Image.new("RGB",(1024,560),"white");draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f"{identifier}: {cases[row['case']]['target']}",fill="black")
        draw.text((10,28),"Original",fill="black");draw.text((522,28),"Edited",fill="black")
        for x,path in [(0,source),(512,target)]:
            if path.exists():
                with Image.open(path) as picture: canvas.paste(picture.convert("RGB"),(x,48))
            else: draw.text((x+10,70),"MISSING/FAILED",fill="red")
        canvas.save(directory/f"{identifier}.png")
        ratings.append({"review_id":identifier,"target_instruction":cases[row["case"]]["target"]})
        mapping.append({"review_id":identifier,"case":row["case"],"seed":row["seed"],"variant":row["variant"]})
    fields=["review_id","target_instruction","reviewer_id","source_matches_prompt","whole_object_recolored",
            "shape_retained","boundary_artifacts","extra_objects","background_spillover","notes"]
    with (directory/"ratings.csv").open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(ratings)
    (ROOT/"reports/frozen_review_mapping.json").write_text(json.dumps(mapping,indent=2),encoding="utf-8")
    (directory/"README.md").write_text(
        "# Independent review\n\nMethods are omitted from the paired images and rating rows. "
        "Do not inspect the mapping or quantitative method results before rating. "
        "For each boolean field enter yes/no/uncertain; leave reviewer_id identifiable. "
        "Whole-object recoloring means all intended painted parts, allowing naturally dark/white/nonpainted parts. "
        "Shape retention concerns silhouette, placement, size and part structure. "
        "Boundary artifacts include seams, broken handles, doubled contours or remnants. "
        "Extra objects and background spillover are separate failures. "
        "Keep missing/failed outputs and note them.\n\n"
        "The sheet is intentionally blank: no independent human ratings have been obtained. "
        "This pairing/randomization does not make the project lead or implementing assistant an independent evaluator.\n",
        encoding="utf-8")
    print(f"Prepared {len(ratings)} review pairs; independent ratings remain pending")


if __name__ == "__main__":main()
