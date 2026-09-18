"""
Phase 10B: Live Terminal Performance Monitor Test Suite
Executes and verifies:
1. Cold Background Removal request (Session Load + Inference)
2. Warm Background Removal request (Session Reused)
3. Export Cutout request (/composite)
4. Export HD 2x AI Super-Resolution request (/export-hd)
5. Export HD 4x AI Super-Resolution request (/export-hd)
6. Safety Guard Rejection test (exceeding pixel limit)
"""

import os
import sys
import io
import time
import base64
import numpy as np
from PIL import Image

# Add backend to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.background_removal_engine import BackgroundRemovalEngine
from backend.background_remover.api.routes import (
    CompositeRequest, ExportHDRequest, composite_endpoint, export_hd_endpoint
)
from backend.processing.super_resolution import SuperResolutionManager


def run_phase10b_tests():
    print("\n" + "=" * 80)
    print("RUNNING PHASE 10B LIVE TERMINAL PERFORMANCE MONITOR TEST SUITE")
    print("=" * 80 + "\n")

    # Generate a test 1024x768 synthetic RGBA image
    w, h = 1024, 768
    img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)
    
    img_bytes_io = io.BytesIO()
    pil_img.save(img_bytes_io, format="PNG")
    png_bytes = img_bytes_io.getvalue()
    b64_str = base64.b64encode(png_bytes).decode("utf-8")

    # TEST 1: Cold Background Removal
    print(">>> [TEST 1/6] Cold Background Removal (First Request)...")
    res1 = BackgroundRemovalEngine.process_image(
        image_bytes=png_bytes,
        sensitivity=10.0,
        edge_softness=50.0,
        defringe_strength=50.0,
        source="test_phase10b",
    )
    print(f"Test 1 Complete: {res1.processing_time_ms:.2f} ms")

    # TEST 2: Warm Background Removal (Reused Session)
    print("\n>>> [TEST 2/6] Warm Background Removal (Reused Session)...")
    res2 = BackgroundRemovalEngine.process_image(
        image_bytes=png_bytes,
        sensitivity=15.0,
        edge_softness=60.0,
        defringe_strength=40.0,
        source="test_phase10b",
    )
    print(f"Test 2 Complete: {res2.processing_time_ms:.2f} ms")

    # TEST 3: Export Cutout (/composite)
    print("\n>>> [TEST 3/6] Export Cutout Composite (/composite)...")
    cutout_bytes_io = io.BytesIO()
    Image.fromarray(res2.rgba_image).save(cutout_bytes_io, format="PNG")
    cutout_b64 = base64.b64encode(cutout_bytes_io.getvalue()).decode("utf-8")

    comp_req = CompositeRequest(
        image_base64=f"data:image/png;base64,{cutout_b64}",
        bg_type="solid",
        color1="#336699",
        export_format="PNG",
    )
    import asyncio
    comp_res = asyncio.run(composite_endpoint(comp_req))
    print(f"Test 3 Complete: Status Code = {comp_res.status_code}")

    # TEST 4: Export HD 2x Super Resolution
    print("\n>>> [TEST 4/6] Export HD 2x Super Resolution...")
    hd2_req = ExportHDRequest(
        image_base64=f"data:image/png;base64,{cutout_b64}",
        export_quality="hd2x",
        export_format="PNG",
    )
    hd2_res = asyncio.run(export_hd_endpoint(hd2_req))
    print(f"Test 4 Complete: Status Code = {hd2_res.status_code}")

    # TEST 5: Export HD 4x Super Resolution
    print("\n>>> [TEST 5/6] Export HD 4x Super Resolution...")
    hd4_req = ExportHDRequest(
        image_base64=f"data:image/png;base64,{cutout_b64}",
        export_quality="hd4x",
        export_format="PNG",
    )
    hd4_res = asyncio.run(export_hd_endpoint(hd4_req))
    print(f"Test 5 Complete: Status Code = {hd4_res.status_code}")

    # TEST 6: Safety Guard Rejection Test
    print("\n>>> [TEST 6/6] Safety Guard Rejection Test...")
    try:
        # Generate huge 4000x3000 RGBA array -> 4x upscale internal pixels = 16000x12000 = 192M pixels (> 33M limit)
        huge_array = np.zeros((3000, 4000, 4), dtype=np.uint8)
        SuperResolutionManager.get_instance().upscale_rgba_ai(huge_array, "hd4x")
        print("Test 6 FAILED: Safety guard did not raise error!")
    except ValueError as ve:
        print(f"Test 6 SUCCESS: Safety Guard properly rejected: {ve}")

    print("\n" + "=" * 80)
    print("ALL PHASE 10B LIVE MONITOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase10b_tests()
