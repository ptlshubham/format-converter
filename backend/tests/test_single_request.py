import os
import sys
import json
import urllib.request
import urllib.error

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
img_path = os.path.join(project_dir, "test_data", "test_portrait_images", "portrait_im", "img_1585.png")
if not os.path.exists(img_path):
    img_path = os.path.join(project_dir, "Solo_Man_Festive_Background.png")

with open(img_path, "rb") as f:
    img_bytes = f.read()

boundary = "----Boundary123"
body = bytearray()
body.extend(f"--{boundary}\r\n".encode())
body.extend(f'Content-Disposition: form-data; name="file"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
body.extend(img_bytes)
body.extend(b"\r\n")
for k, v in [("sensitivity", "10.0"), ("edge_softness", "50.0"), ("defringe_strength", "50.0")]:
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
body.extend(f"--{boundary}--\r\n".encode())

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/background-remover/remove",
    data=bytes(body),
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
        print("SUCCESS! Status:", resp.status)
        print("Stage timings:", data.get("stageTimings"))
        print("Diagnostics:", data.get("diagnostics"))
except urllib.error.HTTPError as e:
    print("HTTPError code:", e.code)
    print("HTTPError body:", e.read().decode())
