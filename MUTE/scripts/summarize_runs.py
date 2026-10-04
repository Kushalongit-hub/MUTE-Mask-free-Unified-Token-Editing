"""Summarize completed development runs without interpreting edit quality."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = {"scope": "Exploratory generated-image development checks",
               "warning": "Inferred-mask background MAE does not establish edit success or independent preservation",
               "runs": []}
    for directory in args.runs:
        data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if data["status"] != "complete":
            raise ValueError(f"Incomplete run: {directory}")
        record = {"directory": str(directory), "seed": data["config"]["seed"],
                  "config": data["config"],
                  "capture_verification": data.get("capture_verification"),
                  "active_patches": data["localization"]["active_patches"],
                  "patches": data["localization"]["patches"], "modes": {}}
        for mode in data["config"]["modes"]:
            for suffix in ["", "_identity", "_restored", "_restored_identity"]:
                name = mode + suffix
                if name in data["runs"]:
                    record["modes"][name] = data["runs"][name]
        if "trajectory" in data:
            record["trajectory"] = data["trajectory"]
        if "restoration_mask" in data:
            record["restoration_mask"] = data["restoration_mask"]
        summary["runs"].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
