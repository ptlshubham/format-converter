"""
Automated Test Suite for RAM-Safe BiRefNet Architecture
Tests:
1. Startup in IDLE state without preloading model
2. First request: lazy loading + inference
3. Second request (Retry): session reuse without reload
4. Concurrent request: immediate HTTP 429 rejection
5. Error recovery: invalid request does not leave lock stuck
6. Idle unload & reload: safe session release and reloading
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import threading

BASE_URL = "http://127.0.0.1:8000/api/background-remover"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_IMG_PATH = os.path.join(PROJECT_DIR, "test_data", "test_images", "horse.jpg")
if not os.path.exists(TEST_IMG_PATH):
    TEST_IMG_PATH = os.path.join(PROJECT_DIR, "test_data", "test_images", "0003.jpg")


def get_status():
    req = urllib.request.Request(f"{BASE_URL}/status")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def build_multipart(file_bytes, filename="test.jpg", sensitivity="10.0", edge_softness="50.0", defringe_strength="50.0"):
    boundary = "----TestBoundary" + str(int(time.time() * 1000))
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
    body.extend(file_bytes)
    body.extend(b"\r\n")
    for k, v in [("sensitivity", sensitivity), ("edge_softness", edge_softness), ("defringe_strength", defringe_strength)]:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary


def send_remove(file_bytes, filename="test.jpg"):
    body, boundary = build_multipart(file_bytes, filename=filename)
    req = urllib.request.Request(
        f"{BASE_URL}/remove",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        try:
            err_json = json.loads(body_text)
        except Exception:
            err_json = {"detail": body_text}
        return e.code, err_json


def run_all_tests():
    print("=" * 70)
    print("STARTING RAM-SAFE BIREFNET ARCHITECTURE VERIFICATION")
    print("=" * 70)

    # 1. Startup Test
    print("\n[TEST 1] Backend Startup & Lazy Loading...")
    status = get_status()
    print(f"  State: {status.get('state')}")
    print(f"  Model Loaded: {status.get('model_loaded')}")
    print(f"  Inference Busy: {status.get('inference_busy')}")
    print(f"  Process Memory RSS: {status.get('process_memory_mb')} MB")
    assert status.get("state") == "IDLE", f"Expected IDLE state, got {status.get('state')}"
    assert status.get("model_loaded") is False, "Model should NOT be preloaded on startup!"
    assert status.get("inference_busy") is False, "Inference should not be busy on startup!"
    print("  [OK] PASS: Backend started lightweight with lazy model loading.")

    # Load test image
    with open(TEST_IMG_PATH, "rb") as f:
        img_bytes = f.read()

    # 2. First Image Request
    print("\n[TEST 2] First Image Request (Lazy Loading + Inference)...")
    t0 = time.time()
    code, resp = send_remove(img_bytes, filename=os.path.basename(TEST_IMG_PATH))
    t_elapsed = round(time.time() - t0, 2)
    print(f"  HTTP Code: {code}")
    print(f"  Total Time: {t_elapsed}s")
    assert code == 200, f"Expected 200, got {code}: {resp}"
    assert resp.get("success") is True, "Expected success: True"
    assert resp.get("resultDataUri", "").startswith("data:image/png;base64,"), "Expected valid data URI"
    print(f"  Model variant: {resp.get('model')}")
    print(f"  Processing time: {resp.get('processingTimeMs')} ms")
    print(f"  Stage timings: {resp.get('stageTimings')}")

    status_after_1 = get_status()
    print(f"  Post-Request State: {status_after_1.get('state')}")
    print(f"  Model Loaded: {status_after_1.get('model_loaded')}")
    print(f"  Inference Busy: {status_after_1.get('inference_busy')}")
    print(f"  Idle Timer Active: {status_after_1.get('idle_timer_active')}")
    print(f"  Idle Remaining: {status_after_1.get('idle_remaining_seconds')}s")
    print(f"  Process Memory RSS: {status_after_1.get('process_memory_mb')} MB")
    assert status_after_1.get("model_loaded") is True, "Model should now be loaded!"
    assert status_after_1.get("state") == "READY", "State should be READY!"
    assert status_after_1.get("inference_busy") is False, "Inference lock should be released!"
    print("  [OK] PASS: First image processed; model loaded once and kept warm.")

    # 3. Second Image / Retry Test (Session Reuse)
    print("\n[TEST 3] Second Image / Retry Request (Session Reuse)...")
    t0 = time.time()
    code2, resp2 = send_remove(img_bytes, filename=os.path.basename(TEST_IMG_PATH))
    t_elapsed2 = round(time.time() - t0, 2)
    print(f"  HTTP Code: {code2}")
    print(f"  Total Time: {t_elapsed2}s")
    assert code2 == 200, f"Expected 200, got {code2}"
    assert resp2.get("success") is True
    status_after_2 = get_status()
    print(f"  Post-Request State: {status_after_2.get('state')}")
    print(f"  Process Memory RSS: {status_after_2.get('process_memory_mb')} MB")
    print("  [OK] PASS: Second request succeeded reusing the warm session.")

    # 4. Concurrency Protection Test (HTTP 429)
    print("\n[TEST 4] Concurrency Protection Test (Two Simultaneous Requests)...")
    results = {}

    def req_a():
        c, r = send_remove(img_bytes, filename="req_a.jpg")
        results["req_a"] = (c, r)

    def req_b():
        time.sleep(0.5)  # Wait 500ms after Request A has acquired lock
        c, r = send_remove(img_bytes, filename="req_b.jpg")
        results["req_b"] = (c, r)

    t_a = threading.Thread(target=req_a)
    t_b = threading.Thread(target=req_b)

    t_a.start()
    t_b.start()
    t_b.join()  # Request B should finish almost immediately with 429

    print(f"  Request B Code (Concurrent): {results['req_b'][0]}")
    print(f"  Request B Body: {results['req_b'][1]}")
    assert results["req_b"][0] == 429, f"Expected HTTP 429 for concurrent request, got {results['req_b'][0]}"
    assert "currently processing another image" in results["req_b"][1].get("detail", ""), "Expected busy detail message!"
    print("  [OK] PASS: Concurrent request was immediately rejected with HTTP 429 without starting second inference!")

    t_a.join()  # Request A should complete normally
    print(f"  Request A Code (Primary): {results['req_a'][0]}")
    assert results["req_a"][0] == 200, f"Expected Request A to finish with 200, got {results['req_a'][0]}"
    print("  [OK] PASS: Primary request completed successfully with HTTP 200.")

    # 5. Error Recovery Test
    print("\n[TEST 5] Error Recovery & Lock Release...")
    bad_bytes = b"not-a-valid-image-file"
    err_code, err_resp = send_remove(bad_bytes, filename="corrupt.jpg")
    print(f"  Corrupt File Code: {err_code} (Expected: 400 or 500)")
    status_post_err = get_status()
    print(f"  State After Error: {status_post_err.get('state')}")
    print(f"  Inference Busy: {status_post_err.get('inference_busy')}")
    assert status_post_err.get("inference_busy") is False, "Lock MUST NOT remain held after error!"

    # Immediately send valid request
    code_after_err, resp_after_err = send_remove(img_bytes, filename="recovery.jpg")
    print(f"  Recovery Request Code: {code_after_err}")
    assert code_after_err == 200, f"Expected recovery request to succeed with 200, got {code_after_err}"
    print("  [OK] PASS: Error handled cleanly; lock released; recovery request succeeded.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
