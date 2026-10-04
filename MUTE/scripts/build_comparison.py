"""Build a compact contact sheet from completed, unmodified run images."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["source", "none", "directional",
                                                      "none_restored", "directional_restored"])
    args = parser.parse_args()
    sheet = Image.new("RGB", (256 * len(args.modes), 300 * len(args.runs)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, directory in enumerate(args.runs):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run: {directory}")
        draw.text((8, row * 300 + 4), directory.name, fill="black")
        for column, mode in enumerate(args.modes):
            with Image.open(directory / f"{mode}.png") as picture:
                sheet.paste(picture.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS),
                            (column * 256, row * 300 + 44))
            draw.text((column * 256 + 8, row * 300 + 25), mode, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
