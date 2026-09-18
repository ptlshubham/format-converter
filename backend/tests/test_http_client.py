import io
import urllib.request
import numpy as np
from PIL import Image

# Create a small 64x64 test image
img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format="PNG")
data = buf.getvalue()

boundary = "----WebKitFormBoundaryTest123"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode() + data + (
    f"\r\n--{boundary}\r\n"
    f'Content-Disposition: form-data; name="sensitivity"\r\n\r\n10\r\n'
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="edge_softness"\r\n\r\n50\r\n'
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="defringe_strength"\r\n\r\n50\r\n'
    f"--{boundary}--\r\n"
).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/background-remover/remove",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

try:
    with urllib.request.urlopen(req) as resp:
        print("STATUS:", resp.status)
        res_bytes = resp.read()
        print("RESPONSE LENGTH:", len(res_bytes))
except Exception as e:
    print("ERROR:", e)
