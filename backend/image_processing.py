"""
Image Processing Utilities: Format Conversion, Compositing, and Safe Validation
Handles EXIF rotation, safe dimensions validation, background compositing (solid/gradient/image),
and export encoding (PNG, WEBP, JPG).
"""

import io
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageOps

# Allow processing of high-resolution images safely while protecting against memory exhaustion
Image.MAX_IMAGE_PIXELS = 100_000_000  # Up to 100 Megapixels safe ceiling

# Safe Input Limits
MAX_UPLOAD_BYTES = 60 * 1024 * 1024       # 60 MB file size limit
MAX_DIMENSION = 8192                      # 8192px width or height limit
MAX_TOTAL_PIXELS = 45_000_000             # 45 Million pixels (~6700x6700)


def get_process_memory_mb() -> float:
    """
    Returns current process Resident Set Size / Working Set in Megabytes.
    Uses psutil if available, otherwise falls back to Windows K32GetProcessMemoryCounters via ctypes.
    """
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.kernel32.K32GetProcessMemoryCounters(handle, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    return 0.0


def get_system_memory_info() -> Dict[str, Any]:
    """
    Returns read-only Windows system physical memory and commit/paging metrics.
    Safe and zero-dependency using ctypes GlobalMemoryStatusEx.
    """
    try:
        import ctypes
        from ctypes import wintypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return {
                "memory_load_pct": int(stat.dwMemoryLoad),
                "total_phys_mb": round(stat.ullTotalPhys / (1024.0 * 1024.0), 1),
                "avail_phys_mb": round(stat.ullAvailPhys / (1024.0 * 1024.0), 1),
                "used_phys_mb": round((stat.ullTotalPhys - stat.ullAvailPhys) / (1024.0 * 1024.0), 1),
                "total_pagefile_mb": round(stat.ullTotalPageFile / (1024.0 * 1024.0), 1),
                "avail_pagefile_mb": round(stat.ullAvailPageFile / (1024.0 * 1024.0), 1),
                "commit_used_mb": round((stat.ullTotalPageFile - stat.ullAvailPageFile) / (1024.0 * 1024.0), 1),
            }
    except Exception:
        pass
    return {
        "memory_load_pct": 0,
        "total_phys_mb": 0.0,
        "avail_phys_mb": 0.0,
        "used_phys_mb": 0.0,
        "total_pagefile_mb": 0.0,
        "avail_pagefile_mb": 0.0,
        "commit_used_mb": 0.0,
    }


def validate_and_decode_image(image_bytes: bytes) -> Tuple[np.ndarray, str, int, int]:
    """
    Safely validates file size and image dimensions to prevent memory exhaustion,
    applies EXIF rotation, and returns RGB numpy array along with format and dimensions.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Uploaded image file is empty.")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"Uploaded file exceeds safe maximum limit of {max_mb}MB.")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to decode image data: {str(e)}")

    original_format = (pil_image.format or "PNG").upper()

    # Apply EXIF orientation
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass

    w, h = pil_image.size
    total_px = w * h

    # Check bounds
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        raise ValueError(
            f"Image dimensions ({w}x{h}) exceed maximum allowed dimension limit of {MAX_DIMENSION}px."
        )

    if total_px > MAX_TOTAL_PIXELS:
        raise ValueError(
            f"Total pixel count ({total_px:,} pixels) exceeds safe limit of {MAX_TOTAL_PIXELS:,} pixels."
        )

    # Convert to RGB uint8
    rgb_image = pil_image.convert("RGB")
    rgb_array = np.array(rgb_image, dtype=np.uint8)

    return rgb_array, original_format, w, h


def export_rgba_bytes(
    rgba_image: np.ndarray,
    export_format: str = "PNG",
    quality: int = 92
) -> bytes:
    """
    Encodes RGBA image into PNG, WEBP, or JPG bytes.
    For JPG, transparency is blended onto a clean white background.
    """
    fmt = export_format.upper().strip()
    if fmt == "JPEG":
        fmt = "JPG"

    h, w, c = rgba_image.shape
    if c != 4:
        raise ValueError(f"Expected RGBA image with 4 channels, got {c}")

    rgb = rgba_image[:, :, :3]
    alpha = rgba_image[:, :, 3]

    if fmt == "JPG":
        # Alpha blend onto white background for JPG export
        alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]
        white_bg = np.full((h, w, 3), 255, dtype=np.float32)
        composite_rgb = np.clip(rgb.astype(np.float32) * alpha_f + white_bg * (1.0 - alpha_f), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(composite_rgb, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    elif fmt == "WEBP":
        pil_img = Image.fromarray(rgba_image, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", quality=quality, lossless=False)
        return buf.getvalue()

    else:  # Default PNG (lossless true RGBA)
        pil_img = Image.fromarray(rgba_image, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def composite_background(
    rgba_image: np.ndarray,
    bg_type: str = "transparent",
    color1: str = "#FFFFFF",
    color2: Optional[str] = "#000000",
    gradient_direction: str = "to-bottom",
    bg_image_bytes: Optional[bytes] = None,
) -> np.ndarray:
    """
    Composites the foreground RGBA image onto a selected replacement background:
    - transparent
    - solid color
    - linear gradient
    - custom background image
    """
    if bg_type == "transparent":
        return rgba_image

    h, w = rgba_image.shape[:2]
    fg_rgb = rgba_image[:, :, :3].astype(np.float32)
    alpha = (rgba_image[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        h_str = hex_str.lstrip("#")
        if len(h_str) == 3:
            h_str = "".join([c * 2 for c in h_str])
        return tuple(int(h_str[i:i + 2], 16) for i in (0, 2, 4))

    bg_rgb = np.full((h, w, 3), 255, dtype=np.float32)

    if bg_type == "solid":
        r, g, b = hex_to_rgb(color1 or "#FFFFFF")
        bg_rgb[:] = [r, g, b]

    elif bg_type == "gradient":
        r1, g1, b1 = hex_to_rgb(color1 or "#6366F1")
        r2, g2, b2 = hex_to_rgb(color2 or "#EC4899")

        if gradient_direction == "to-right":
            t = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :, np.newaxis]
            bg_rgb = (1.0 - t) * [r1, g1, b1] + t * [r2, g2, b2]
            bg_rgb = np.repeat(bg_rgb, h, axis=0)
        else:  # to-bottom
            t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis, np.newaxis]
            bg_rgb = (1.0 - t) * [r1, g1, b1] + t * [r2, g2, b2]
            bg_rgb = np.repeat(bg_rgb, w, axis=1)

    elif bg_type == "image" and bg_image_bytes:
        try:
            bg_pil = Image.open(io.BytesIO(bg_image_bytes)).convert("RGB")
            bg_resized = bg_pil.resize((w, h), Image.BILINEAR)
            bg_rgb = np.array(bg_resized, dtype=np.float32)
        except Exception:
            pass

    # Standard Alpha Blend: FG * alpha + BG * (1 - alpha)
    composite_f = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    composite_uint8 = np.clip(composite_f, 0, 255).astype(np.uint8)

    # Return as RGBA with full opacity
    full_alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([composite_uint8, full_alpha], axis=2)
