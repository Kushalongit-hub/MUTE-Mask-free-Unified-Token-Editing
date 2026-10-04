"""Resumable verified HTTP range download when large Hub transfers stall."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import model_info
import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--chunk-mb", type=int, default=8)
    p.add_argument("--revision", default="50f702106901db6d0f8b67eb88e814c56ded2692")
    args = p.parse_args()
    model = "PixArt-alpha/PixArt-XL-2-512x512"
    info = model_info(model, revision=args.revision, files_metadata=True)
    wanted = [s for s in info.siblings if
        s.rfilename == "model_index.json" or s.rfilename.endswith("config.json") or
        s.rfilename.startswith("tokenizer/") or
        s.rfilename.startswith("text_encoder/model.fp16-") or
        s.rfilename == "text_encoder/model.safetensors.index.fp16.json" or
        s.rfilename == "transformer/diffusion_pytorch_model.safetensors" or
        s.rfilename == "vae/diffusion_pytorch_model.fp16.safetensors"]
    args.output.mkdir(parents=True, exist_ok=True)
    chunks = args.output / ".chunks"
    chunks.mkdir(exist_ok=True)
    provenance = {"model": model, "revision": info.sha, "files": {}}
    for item in wanted:
        target = args.output / item.rfilename
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{model}/resolve/{info.sha}/{item.rfilename}"
        expected_hash = item.lfs.sha256 if item.lfs else None
        if target.exists() and target.stat().st_size == item.size:
            digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
            if not expected_hash or digest == expected_hash:
                provenance["files"][item.rfilename] = {"size": item.size, "sha256": digest}
                print("cached", item.rfilename, flush=True)
                continue
        if item.size < 4 * 1024 * 1024:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            target.write_bytes(r.content)
        else:
            chunk_size = args.chunk_mb * 1024 * 1024
            parts = []
            prefix = hashlib.sha256(item.rfilename.encode()).hexdigest()[:16]
            for index, start in enumerate(range(0, item.size, chunk_size)):
                end = min(start + chunk_size, item.size) - 1
                parts.append((chunks / f"{prefix}_{index:05d}", start, end))

            def download(part):
                path, start, end = part
                if path.exists() and path.stat().st_size == end - start + 1:
                    return
                for attempt in range(6):
                    try:
                        with requests.get(url + f"?download=true&mute_chunk={start}",
                                headers={"Range": f"bytes={start}-{end}"},
                                stream=True, timeout=(30, 60)) as response:
                            response.raise_for_status()
                            expected = f"bytes {start}-{end}/{item.size}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                                raise ValueError("Server did not honor the exact requested byte range")
                            temp = path.with_suffix(".partial")
                            with temp.open("wb") as out:
                                for block in response.iter_content(1024 * 1024):
                                    out.write(block)
                            if temp.stat().st_size != end - start + 1:
                                raise ValueError("Incomplete chunk")
                            temp.replace(path)
                            return
                    except (requests.RequestException, ValueError):
                        if attempt == 5:
                            raise
                        time.sleep(min(2 ** attempt, 16))

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download, part) for part in parts]
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if count % 16 == 0 or count == len(parts):
                        print(f"{item.rfilename}: {count}/{len(parts)} chunks", flush=True)
            temp = target.with_suffix(target.suffix + ".partial")
            with temp.open("wb") as out:
                for path, _, _ in parts:
                    with path.open("rb") as src:
                        while block := src.read(4 * 1024 * 1024):
                            out.write(block)
            target = temp
        with target.open("rb") as src:
            digest = hashlib.file_digest(src, "sha256").hexdigest()
        if target.stat().st_size != item.size or (expected_hash and digest != expected_hash):
            raise ValueError(f"Integrity verification failed: {item.rfilename}")
        if target.name.endswith(".partial"):
            target.replace(args.output / item.rfilename)
        provenance["files"][item.rfilename] = {"size": item.size, "sha256": digest}
        (args.output / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print("verified", item.rfilename, flush=True)
    print("Checkpoint complete", flush=True)


if __name__ == "__main__":
    main()
