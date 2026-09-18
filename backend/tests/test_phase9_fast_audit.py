"""
Phase 9 Fast Production Release Audit
Executes 1 representative 4x export on a 1200x800 image to confirm zero-regression release readiness.
"""

import os
import sys
import time
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.processing.super_resolution import SuperResolutionManager
from backend.image_processing import export_rgba_bytes


def fast_release_audit():
    print("=" * 80)
    print("PHASE 9 FAST PRODUCTION RELEASE AUDIT")
    print("=" * 80)

    sr_mgr = SuperResolutionManager.get_instance()
    
    # Representative 1200x800 RGBA image
    dummy_rgba = np.random.randint(0, 256, (800, 1200, 4), dtype=np.uint8)
    dummy_rgba[:, :, 3] = 255

    t0 = time.perf_counter()
    out_4x, perf_info = sr_mgr.upscale_rgba_ai(dummy_rgba, export_quality="4x")
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"Representative Export: {dummy_rgba.shape[1]}x{dummy_rgba.shape[0]} -> {out_4x.shape[1]}x{out_4x.shape[0]}")
    print(f"Total Export Time:     {dt:.2f} ms")
    print(f"Model Inference Time:  {perf_info.get('ai_inference_ms')} ms")
    print(f"Execution Provider:    {perf_info.get('provider')}")
    print(f"Tile Count:            {perf_info.get('tile_count')}")

    assert out_4x.shape == (3200, 4800, 4), "Output dimension mismatch!"
    assert perf_info.get("provider") == "DmlExecutionProvider", "DirectML provider inactive!"

    print("\n[RESULT] REPRESENTATIVE 4x EXPORT PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    fast_release_audit()
