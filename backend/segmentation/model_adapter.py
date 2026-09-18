"""
Model Adapter Architecture for AI Segmentation
BiRefNet General FP16 ONNX model adapter with standardized preprocessing and postprocessing.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from ..image_processing import get_process_memory_mb, get_system_memory_info

logger = logging.getLogger("BackgroundRemoval.Adapter")


class SegmentationModel(ABC):
    """
    Abstract Base Class for Background Segmentation Models.
    Enforces a uniform interface across all AI architectures.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the human-readable name of the model."""
        pass

    @property
    @abstractmethod
    def license(self) -> str:
        """Returns the model's official software license."""
        pass

    @property
    @abstractmethod
    def is_commercial_friendly(self) -> bool:
        """Returns True if the license permits commercial usage."""
        pass

    @property
    @abstractmethod
    def input_resolution(self) -> Tuple[int, int]:
        """Returns the native input dimensions (width, height)."""
        pass

    @abstractmethod
    def predict_probability_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Takes an RGB uint8 image of any dimension and returns a continuous
        foreground probability map (float32, values between 0.0 and 1.0)
        matching the original image dimensions.
        """
        pass


class BiRefNetAdapter(SegmentationModel):
    """
    Adapter for BiRefNet (Bilateral Reference Network for High-Resolution Dichotomous Image Segmentation).
    - License: MIT License (Commercial Friendly)
    - Input: 1024x1024 RGB tensor with letterbox padding & ImageNet normalization
    - Output: Raw logits mapped through continuous float32 sigmoid activation, un-letterboxed to native resolution.
    """

    def __init__(self, session: Any, model_variant: str = "general"):
        self.session = session
        self.model_variant = model_variant.lower()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Precomputed ImageNet constants for fast in-place vector math
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.inv_std_255 = (1.0 / (255.0 * self.std)).astype(np.float32)
        self.mean_over_std = (self.mean / self.std).astype(np.float32)
        self.latest_timings: Dict[str, float] = {}

    @property
    def model_name(self) -> str:
        return "BiRefNet General (FP16 ONNX)"

    @property
    def license(self) -> str:
        return "MIT License"

    @property
    def is_commercial_friendly(self) -> bool:
        return True

    @property
    def input_resolution(self) -> Tuple[int, int]:
        return (1024, 1024)

    def warmup(self):
        """
        Warmup is intentionally a no-op.
        The model requires fixed 1024x1024 input and running a full-size tensor
        on CPU causes OOM ('bad allocation'). The first real inference request
        will pay a small one-time initialization cost instead.
        """
        pass

    def predict_probability_map(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Executes optimized BiRefNet inference with aspect-ratio preserving letterbox:
        1. Fast scaling factor and area/linear interpolation.
        2. Centered letterbox padding with pre-cached constant.
        3. Direct single-pass normalization: x * inv_std_255 - mean_over_std.
        4. Transpose to contiguous float32 NCHW [1, 3, 1024, 1024].
        5. ONNX inference using cached session.
        6. In-place sigmoid calculation and bilinear restore to exact native resolution.
        """
        orig_h, orig_w = rgb_image.shape[:2]
        target_size = 1024

        t_pre = time.perf_counter()
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        interp = cv2.INTER_AREA if (new_w < orig_w or new_h < orig_h) else cv2.INTER_LINEAR
        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=interp)

        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left

        padded_img = np.full((target_size, target_size, 3), (124, 116, 104), dtype=np.uint8)
        padded_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized

        # Single-pass standardized tensor creation: x * inv_std_255 - mean_over_std
        norm_img = padded_img.astype(np.float32) * self.inv_std_255 - self.mean_over_std
        tensor = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)
        pre_ms = round((time.perf_counter() - t_pre) * 1000.0, 2)

        # PHASE 2: Diagnostic Profiling Immediately BEFORE session.run()
        mem_before = get_process_memory_mb()
        sys_before = get_system_memory_info()
        inp_shape = list(tensor.shape)
        inp_dtype = str(tensor.dtype)
        inp_mb = round(tensor.nbytes / (1024.0 * 1024.0), 2)
        pid = os.getpid()

        logger.info("-" * 60)
        logger.info("[PROFILE] BEFORE ONNX INFERENCE")
        logger.info(f"PID:          {pid}")
        logger.info(f"Process RSS:  {mem_before} MB")
        logger.info(f"System RAM:   {sys_before['memory_load_pct']}% (Used: {sys_before['used_phys_mb']} MB / Total: {sys_before['total_phys_mb']} MB)")
        logger.info(f"Avail RAM:    {sys_before['avail_phys_mb']} MB")
        logger.info(f"Commit/Page:  Used {sys_before['commit_used_mb']} MB / Limit {sys_before['total_pagefile_mb']} MB")
        logger.info(f"Provider:     CPUExecutionProvider")
        logger.info(f"Threads:      4 intra-op, 1 inter-op")
        logger.info(f"Input shape:  {inp_shape}")
        logger.info(f"Input dtype:  {inp_dtype}")
        logger.info(f"Input memory: {inp_mb} MB")
        logger.info("-" * 60)

        # Run ONNX inference
        t_onnx = time.perf_counter()
        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})
        onnx_ms = round((time.perf_counter() - t_onnx) * 1000.0, 2)

        # PHASE 2: Diagnostic Profiling Immediately AFTER session.run()
        mem_after = get_process_memory_mb()
        sys_after = get_system_memory_info()
        out_tensor = raw_outputs[0]
        out_shape = list(out_tensor.shape)
        out_dtype = str(out_tensor.dtype)
        out_mb = round(out_tensor.nbytes / (1024.0 * 1024.0), 2)
        out_min = round(float(np.min(out_tensor)), 4)
        out_max = round(float(np.max(out_tensor)), 4)
        out_contiguous = bool(out_tensor.flags.c_contiguous)

        logger.info("-" * 60)
        logger.info("[PROFILE] AFTER ONNX INFERENCE")
        logger.info(f"Inference Time: {onnx_ms} ms ({round(onnx_ms/1000.0, 2)} s)")
        logger.info(f"Process RSS:    {mem_after} MB (Delta: +{round(mem_after - mem_before, 2)} MB)")
        logger.info(f"System RAM:     {sys_after['memory_load_pct']}% (Used: {sys_after['used_phys_mb']} MB)")
        logger.info(f"Avail RAM:      {sys_after['avail_phys_mb']} MB")
        logger.info(f"Commit/Page:    Used {sys_after['commit_used_mb']} MB")
        logger.info(f"Output count:   {len(raw_outputs)}")
        logger.info(f"Output shape:   {out_shape}")
        logger.info(f"Output dtype:   {out_dtype}")
        logger.info(f"Output memory:  {out_mb} MB")
        logger.info(f"Output min/max: [{out_min}, {out_max}]")
        logger.info(f"Contiguous:     {out_contiguous}")
        logger.info("-" * 60)

        self.latest_profile = {
            "inference_ms": onnx_ms,
            "rss_before_mb": mem_before,
            "rss_after_mb": mem_after,
            "rss_delta_mb": round(mem_after - mem_before, 2),
            "sys_ram_pct_before": sys_before["memory_load_pct"],
            "sys_ram_pct_after": sys_after["memory_load_pct"],
            "sys_avail_ram_mb": sys_after["avail_phys_mb"],
            "input_shape": inp_shape,
            "input_dtype": inp_dtype,
            "input_mb": inp_mb,
            "output_shape": out_shape,
            "output_dtype": out_dtype,
            "output_mb": out_mb,
            "output_min": out_min,
            "output_max": out_max,
        }

        logits = raw_outputs[0][0, 0]  # Shape (1024, 1024)

        # Stage 1 Cleanup: Free input tensor and preprocessing buffers
        del tensor, norm_img, padded_img, resized, raw_outputs

        # Sigmoid activation
        t_sig = time.perf_counter()
        clipped_logits = np.clip(logits, -50.0, 50.0)
        prob_1024 = 1.0 / (1.0 + np.exp(-clipped_logits))
        sig_ms = round((time.perf_counter() - t_sig) * 1000.0, 2)

        # Stage 2 Cleanup: Free logits
        del logits, clipped_logits

        # Unletterbox crop and native resolution resize
        t_resize = time.perf_counter()
        prob_cropped = prob_1024[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        prob_orig = cv2.resize(prob_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        prob_final = np.clip(prob_orig, 0.0, 1.0, out=prob_orig).astype(np.float32)
        resize_ms = round((time.perf_counter() - t_resize) * 1000.0, 2)

        # Stage 3 Cleanup: Free unletterbox intermediate arrays
        del prob_1024, prob_cropped

        self.latest_timings = {
            "preprocess_ms": pre_ms,
            "onnx_inference_ms": onnx_ms,
            "sigmoid_ms": sig_ms,
            "mask_resize_ms": resize_ms,
        }
        return prob_final
