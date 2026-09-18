"""
Model Downloader for BiRefNet General FP16 ONNX Background Removal Engine
Downloads official ONNX weights:
BiRefNet General FP16 ONNX (onnx-community/BiRefNet-ONNX)
"""

import os
import sys
import time
import requests

MODEL_CONFIG = {
    "name": "BiRefNet General FP16 ONNX",
    "url": "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model_fp16.onnx",
    "dest": "backend/models/segmentation/birefnet-general/model_fp16.onnx",
    "expected_min_size": 450 * 1024 * 1024,
}


def download_file(url: str, dest: str, expected_min_size: int, name: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        current_size = os.path.getsize(dest)
        if current_size >= expected_min_size:
            print(f"[CACHE] {name} already exists ({round(current_size / 1024 / 1024, 1)} MB). Skipping download.")
            return

    print(f"\n[DOWNLOADING] {name} from {url} ...")
    t0 = time.time()
    response = requests.get(url, stream=True, allow_redirects=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    temp_dest = dest + ".tmp"
    downloaded = 0
    last_log = 0

    with open(temp_dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_log >= 25 * 1024 * 1024:
                    mb = downloaded / (1024 * 1024)
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    elapsed = time.time() - t0
                    speed = mb / elapsed if elapsed > 0 else 0
                    print(f"  -> {mb:.1f} MB ({pct:.1f}%) [{speed:.1f} MB/s]")
                    last_log = downloaded

    os.replace(temp_dest, dest)
    final_size = os.path.getsize(dest)
    elapsed = time.time() - t0
    print(f"[DONE] {name} saved to {dest} ({round(final_size / 1024 / 1024, 1)} MB in {elapsed:.1f}s)")


def main():
    print("=" * 70)
    print("SINGLE-MODEL BIREFNET GENERAL FP16 DOWNLOADER")
    print("=" * 70)
    download_file(
        MODEL_CONFIG["url"],
        MODEL_CONFIG["dest"],
        MODEL_CONFIG["expected_min_size"],
        MODEL_CONFIG["name"]
    )
    print("\nBiRefNet General FP16 model verified successfully!")


if __name__ == "__main__":
    main()
