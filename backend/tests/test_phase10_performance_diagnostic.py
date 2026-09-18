"""
Phase 10: Background Removal Performance Diagnostic Test Script
Executes ONE representative Background Removal request on a 1536x1024 image.
Collects and displays the complete Phase 10 performance profile and bottleneck report.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.background_removal_engine import BackgroundRemovalEngine


def run_phase10_diagnostic():
    print("=" * 80)
    print("RUNNING PHASE 10 BACKGROUND REMOVAL PERFORMANCE DIAGNOSTIC")
    print("=" * 80)

    # Find a test image or create a 1536x1024 input image
    test_image_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_data", "test_human_images"
    )
    
    test_file_path = None
    if os.path.exists(test_image_dir):
        files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            test_file_path = os.path.join(test_image_dir, files[0])

    if test_file_path and os.path.exists(test_file_path):
        print(f"Loading test image: {test_file_path}")
        pil_img = Image.open(test_file_path).convert("RGB")
    else:
        print("Generating synthetic test image...")
        pil_img = Image.fromarray(np.random.randint(0, 256, (1024, 1536, 3), dtype=np.uint8))

    # Resize to exact 1536 x 1024 target test resolution
    pil_1536 = pil_img.resize((1536, 1024), Image.Resampling.LANCZOS)
    
    # Convert to PNG bytes
    import io
    img_byte_arr = io.BytesIO()
    pil_1536.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()
    
    print(f"Input image decoded: {pil_1536.width} × {pil_1536.height} ({len(image_bytes):,} bytes)")
    
    # Process image through Background Removal engine
    result = BackgroundRemovalEngine.process_image(
        image_bytes=image_bytes,
        sensitivity=10.0,
        edge_softness=50.0,
        defringe_strength=50.0,
        source="diagnostic_test"
    )

    print(f"\nProcessing Complete in {result.processing_time_ms:.2f} ms")
    print(f"Result Image Shape: {result.rgba_image.shape}")


if __name__ == "__main__":
    run_phase10_diagnostic()
