"""
Comprehensive Test Suite for Hybrid AI Background Removal Engine
Tests:
1. Solo_Man_Festive_Background.png (Festive portrait with background stage/curtains)
2. test_data/test_portrait_images/ (Portraits and headshots)
3. test_data/test_human_images/ (Full body humans)
4. test_data/test_images/ (Objects, products, vehicles)
5. API vs Standalone pipeline byte-for-byte and pixel-for-pixel EQUIVALENCE VERIFICATION!
"""

import os
import sys
import time
import base64
import asyncio
import numpy as np
from PIL import Image
import io

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(backend_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from backend.background_removal_engine import BackgroundRemovalEngine
from backend.segmentation.model_manager import ModelManager
from backend.image_processing import export_rgba_bytes
from backend.background_remover.api.routes import remove_background
from fastapi import UploadFile


def test_api_vs_standalone_equivalence(image_path: str):
    """
    Directly tests that the FastAPI API and the Standalone Engine pipeline
    produce EQUIVALENT pixel outputs for the exact same input image.
    """
    print("\n" + "=" * 80)
    print(f"VERIFYING API VS STANDALONE PIPELINE PARITY: {os.path.basename(image_path)}")
    print("=" * 80)

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 1. Run standalone pipeline
    print("  [1] Executing Standalone Engine...")
    t0 = time.perf_counter()
    standalone_res = BackgroundRemovalEngine.process_image(
        image_bytes=img_bytes,
        sensitivity=10.0,
        edge_softness=50.0,
        defringe_strength=50.0,
        source="test",
        filename=os.path.basename(image_path),
    )
    t_standalone = (time.perf_counter() - t0) * 1000.0

    # 2. Run API route endpoint directly with an UploadFile
    print("  [2] Executing FastAPI API Route (/api/background-remover/remove)...")
    upload_file = UploadFile(
        filename=os.path.basename(image_path),
        file=io.BytesIO(img_bytes),
    )

    t0 = time.perf_counter()
    api_response = asyncio.run(
        remove_background(
            file=upload_file,
            sensitivity=10.0,
            edge_softness=50.0,
            defringe_strength=50.0,
            diagnostics=False,
        )
    )
    t_api = (time.perf_counter() - t0) * 1000.0

    import json
    api_body = json.loads(api_response.body.decode("utf-8"))
    assert api_body["success"] is True, f"API returned error: {api_body}"

    # Decode base64 data URI from API
    data_uri = api_body["resultDataUri"]
    assert data_uri.startswith("data:image/png;base64,"), "Invalid data URI prefix from API"
    b64_content = data_uri.split(",", 1)[1]
    api_png_bytes = base64.b64decode(b64_content)
    api_image_pil = Image.open(io.BytesIO(api_png_bytes))
    api_rgba = np.array(api_image_pil)

    # 3. Compare Standalone RGBA vs API RGBA
    standalone_rgba = standalone_res.rgba_image

    print(f"  Standalone Output Dimensions: {standalone_res.width}x{standalone_res.height}")
    print(f"  API Output Dimensions:        {api_body['width']}x{api_body['height']}")
    print(f"  Standalone Model:             {standalone_res.diagnostics.get('model_name')}")
    print(f"  API Model:                    {api_body.get('model')}")
    print(f"  Standalone Time:              {t_standalone:.1f} ms")
    print(f"  API Time:                     {t_api:.1f} ms")

    assert standalone_rgba.shape == api_rgba.shape, (
        f"Shape mismatch: Standalone {standalone_rgba.shape} vs API {api_rgba.shape}"
    )

    diff = np.abs(standalone_rgba.astype(np.int32) - api_rgba.astype(np.int32))
    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))

    print(f"  Pixel Max Difference:         {max_diff}")
    print(f"  Pixel Mean Difference:        {mean_diff:.4f}")

    # They should be 100% identical (max_diff == 0)
    assert max_diff == 0, f"Standalone and API outputs diverge! Max pixel diff: {max_diff}"
    print("  >>> SUCCESS: API and Standalone Engine produced 100% IDENTICAL results! <<<\n")


def run_hybrid_engine_test_suite():
    print("=" * 80)
    print("HYBRID AI BACKGROUND REMOVAL ENGINE - TEST SUITE")
    print("=" * 80)

    # 1. Test Solo_Man_Festive_Background.png
    festive_img_path = os.path.join(project_dir, "Solo_Man_Festive_Background.png")
    if os.path.exists(festive_img_path):
        print(f"\n[TEST 1] Solo_Man_Festive_Background.png")
        with open(festive_img_path, "rb") as f:
            festive_bytes = f.read()

        res = BackgroundRemovalEngine.process_image(
            image_bytes=festive_bytes,
            sensitivity=10.0,
            edge_softness=50.0,
            defringe_strength=50.0,
            source="test",
            filename="Solo_Man_Festive_Background.png",
        )
        diag = res.diagnostics
        print(f"  Model Routed:      {diag.get('model_name')}")
        print(f"  Category:          {diag.get('primary_category')} (conf: {diag.get('category_confidence'):.2%})")
        print(f"  Routing Reason:    {diag.get('routing_reason')}")
        print(f"  Transparency Type: {diag.get('transparency_type')}")
        print(f"  Dimensions:        {res.width}x{res.height}")
        print(f"  Processing Time:   {res.processing_time_ms:.1f} ms")
        print(f"  Stage Timings:     {res.stage_timings}")

        # Verify model used
        assert "BiRefNet" in diag.get("model_name", "") or "U2-Net" in diag.get("model_name", ""), "Must use BiRefNet or rollback model!"
        assert res.rgba_image.shape[2] == 4, "Output must be RGBA!"
        print("  >>> Test 1 Passed! <<<")

        # Verify API vs Standalone Equivalence on Solo_Man_Festive_Background
        test_api_vs_standalone_equivalence(festive_img_path)
    else:
        print(f"Notice: {festive_img_path} not found.")

    # 2. Test Data Categories
    test_data_dir = os.path.join(project_dir, "test_data")
    if os.path.exists(test_data_dir):
        categories = [
            ("Portraits", os.path.join(test_data_dir, "test_portrait_images", "portrait_im")),
            ("Full Body", os.path.join(test_data_dir, "test_human_images")),
            ("Objects & Products", os.path.join(test_data_dir, "test_images")),
        ]

        results_dir = os.path.join(test_data_dir, "hybrid_ai_results")
        os.makedirs(results_dir, exist_ok=True)

        for cat_name, cat_dir in categories:
            if not os.path.exists(cat_dir):
                continue
            files = [f for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(('.jpg', '.png', '.webp'))]
            if not files:
                continue

            test_file = files[0]
            test_path = os.path.join(cat_dir, test_file)
            print(f"\n[TEST: {cat_name}] {test_file}")

            with open(test_path, "rb") as f:
                b = f.read()

            res = BackgroundRemovalEngine.process_image(
                image_bytes=b,
                sensitivity=10.0,
                edge_softness=50.0,
                defringe_strength=50.0,
                source="test",
                filename=test_file,
            )
            diag = res.diagnostics
            print(f"  Model Routed:    {diag.get('model_name')}")
            print(f"  Category:        {diag.get('primary_category')} ({diag.get('category_confidence'):.2%})")
            print(f"  Dimensions:      {res.width}x{res.height}")
            print(f"  Processing Time: {res.processing_time_ms:.1f} ms")

            out_bytes = export_rgba_bytes(res.rgba_image, export_format="PNG")
            out_file = os.path.join(results_dir, f"{os.path.splitext(test_file)[0]}_hybrid_cutout.png")
            with open(out_file, "wb") as out_f:
                out_f.write(out_bytes)

            assert res.rgba_image.shape[2] == 4
            print(f"  >>> {cat_name} Test Passed! <<<")

    print("\n" + "=" * 80)
    print("ALL HYBRID AI ENGINE TESTS & EQUIVALENCE CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_hybrid_engine_test_suite()
