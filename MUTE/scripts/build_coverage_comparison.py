"""Reproduce contact sheets for the fixed coverage development sweep."""
import json
from pathlib import Path
from PIL import Image, ImageDraw


def sheet(entries, output):
    canvas = Image.new("RGB", (256 * len(entries), 284), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (directory, mode, label) in enumerate(entries):
        directory = Path("runs") / directory
        if json.loads((directory / "manifest.json").read_text())["status"] != "complete":
            raise ValueError(f"Incomplete run: {directory}")
        with Image.open(directory / f"{mode}.png") as picture:
            canvas.paste(picture.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), (i * 256, 28))
        draw.text((i * 256 + 8, 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    base = "mug_seed42_restoration_v1"
    sheet([(base, "source", "Source"), (base, "none_restored", "Original mask"),
           ("mug_seed42_coverage_m1_f0_v1", "none_restored", "Margin 16 px"),
           ("mug_seed42_coverage_m2_f0_v1", "none_restored", "Margin 32 px"),
           ("mug_seed42_coverage_m2_f1_v1", "none_restored", "Margin 32 px + graded ring")],
          Path("reports/coverage_mug42_2026-10-04.png"))
    for name, seed in [("apple_explicit", 42), ("mug", 123)]:
        base = f"{name}_seed{seed}_restoration_v1"
        entries = [(base, "source", "Source"), (base, "none_restored", "Original mask")]
        if name == "mug":
            entries.append((f"{name}_seed{seed}_coverage_m2_f0_v1", "none_restored", "Margin 32 px"))
        entries.append((f"{name}_seed{seed}_coverage_m2_f1_v1", "none_restored", "Margin 32 px + graded ring"))
        sheet(entries, Path(f"reports/coverage_{name}{seed}_2026-10-04.png"))


if __name__ == "__main__":
    main()
