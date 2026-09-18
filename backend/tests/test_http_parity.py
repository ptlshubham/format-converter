"""
Direct HTTP Network Parity Test
Sends an image over real HTTP socket to http://127.0.0.1:8000/api/background-remover/remove
and compares the resulting decoded RGBA cutout against the standalone engine.
"""

import os
import sys
import json
import base64
import urllib.request
import numpy as np
from PIL import Image
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(backend_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from backend.background_removal_engine import BackgroundRemovalEngine


def run_http_parity_test():
    img_path = os.path.join(project_dir, "test_data", "test_portrait_images", "portrait_im", "img_1585.png")
    if not os.path.exists(img_path):
        # Fallback to another image
        img_path = os.path.join(project_dir, "Solo_Man_Festive_Background.png")

    print(f"Testing HTTP Parity with: {os.path.basename(img_path)}")
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # 1. Standalone Execution
    print("Running Standalone BackgroundRemovalEngine...")
    res_standalone = BackgroundRemovalEngine.process_image(
        image_bytes=img_bytes,
        sensitivity=10.0,
        edge_softness=50.0,
        defringe_strength=50.0,
        source="test",
        filename=os.path.basename(img_path),
    )
    print(f"  Standalone: {res_standalone.width}x{res_standalone.height} | Model: {res_standalone.diagnostics.get('model_name')}")

    # 2. HTTP POST Execution
    print("Sending HTTP POST to http://127.0.0.1:8000/api/background-remover/remove...")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(img_path)}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: image/png\r\n\r\n")
    body.extend(img_bytes)
    body.extend(b"\r\n")

    for k, v in [("sensitivity", "10.0"), ("edge_softness", "50.0"), ("defringe_strength", "50.0")]:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/background-remover/remove",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))

    print(f"  HTTP API:   {resp_data['width']}x{resp_data['height']} | Model: {resp_data.get('model')}")

    data_uri = resp_data["resultDataUri"]
    b64 = data_uri.split(",", 1)[1]
    api_png = base64.b64decode(b64)
    api_img = np.array(Image.open(io.BytesIO(api_png)))

    from backend.image_processing import export_rgba_bytes
    standalone_png = export_rgba_bytes(res_standalone.rgba_image, "PNG")
    standalone_img = np.array(Image.open(io.BytesIO(standalone_png)))
    diff = np.abs(standalone_img.astype(np.int32) - api_img.astype(np.int32))
    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))

    print(f"Pixel Parity Evaluation:")
    print(f"  Max pixel diff:  {max_diff}")
    print(f"  Mean pixel diff: {mean_diff:.4f}")
    assert max_diff <= 1 and mean_diff < 0.05, f"Parity mismatch: max difference {max_diff}"
    print("=" * 60)
    print("SUCCESS: 100% PARITY BETWEEN HTTP API AND STANDALONE ENGINE!")
    print("=" * 60)


if __name__ == "__main__":
    run_http_parity_test()
