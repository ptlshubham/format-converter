"""
Model fetcher for servers without git-lfs.

The repo keeps the *.onnx weights in Git LFS. A plain `git clone` without
git-lfs leaves ~130-byte pointer files in their place, and every request then
fails with a 500. This script downloads the real files from GitHub's public LFS
media endpoint and checks each one against the size and SHA-256 recorded in its
LFS pointer. Files already present with the right hash are skipped, so it is
safe to re-run.

Usage (from any directory; paths are resolved next to this file):
    python backend/models/download_models.py            # fetch anything missing or wrong
    python backend/models/download_models.py --check    # verify only; exit 1 if anything is missing or wrong

Standard library only, so it runs before or after `pip install`.
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://media.githubusercontent.com/media/ptlshubham/format-converter/{ref}/backend/models/{path}"
CHUNK = 4 * 1024 * 1024

# (path under backend/models, bytes, SHA-256 = the oid in the file's Git LFS pointer)
MODELS = [
    # BiRefNet General FP16: ModelManager, required by /remove
    ("segmentation/birefnet-general/model_fp16.onnx", 489_666_272,
     "3654c741eb80bd926ada8fed1713b506ccf8d30eb1f6487e87eb9f234f33df09"),
    # Real-ESRGAN x4plus: SuperResolutionManager, required by /export-hd (2x and 4x)
    ("super_resolution/realesrgan_x4plus.onnx", 67_160_311,
     "6a6f4a3d58553d40fdd443d9e5f4b2deb9b52bef1ec2947700fc2167ac876c7d"),
    # YuNet 2023mar: face preservation during HD export (skipped by the code if missing)
    ("face_detector/face_detection_yunet_2023mar.onnx", 232_589,
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: str, size: int, sha256: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == size and sha256_of(path) == sha256


def download(rel_path: str, size: int, sha256: str, dest: str, ref: str = "main") -> None:
    url = BASE_URL.format(ref=ref, path=rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    print(f"[DOWNLOADING] {rel_path} ({size / 1024 / 1024:.1f} MB) from {url}")

    t0 = time.time()
    digest = hashlib.sha256()
    received = 0
    last_log = 0
    request = urllib.request.Request(url, headers={"User-Agent": "kardlyz-image-ai-model-fetch"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received - last_log >= 50 * 1024 * 1024:
                    print(f"  -> {received / 1024 / 1024:.0f} MB ({received / size * 100:.0f}%)")
                    last_log = received

        if received != size or digest.hexdigest() != sha256:
            raise ValueError(
                f"verification failed: got {received} bytes with SHA-256 {digest.hexdigest()}, "
                f"expected {size} bytes with SHA-256 {sha256}"
            )
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)

    print(f"[DONE] {rel_path} verified ({time.time() - t0:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the image AI service's ONNX models.")
    parser.add_argument("--check", action="store_true", help="verify only, download nothing")
    parser.add_argument("--ref", default="main", help="git branch, tag or commit to fetch from (default: main)")
    args = parser.parse_args()

    failed = False
    for rel_path, size, sha256 in MODELS:
        dest = os.path.join(MODELS_DIR, *rel_path.split("/"))
        if is_valid(dest, size, sha256):
            print(f"[OK] {rel_path}")
            continue
        if args.check:
            state = "missing" if not os.path.exists(dest) else "wrong size or hash (an LFS pointer file?)"
            print(f"[BAD] {rel_path}: {state}")
            failed = True
            continue
        try:
            download(rel_path, size, sha256, dest, ref=args.ref)
        except Exception as e:
            print(f"[FAILED] {rel_path}: {e}", file=sys.stderr)
            failed = True

    if failed:
        print("\nSome models are missing or invalid; the service answers 500 until they are fixed.", file=sys.stderr)
        return 1
    print("\nAll models present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
