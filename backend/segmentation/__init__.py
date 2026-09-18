"""
Segmentation Module: Model Adapters and Model Manager.
"""
from .model_adapter import SegmentationModel, BiRefNetAdapter
from .model_manager import ModelManager

__all__ = [
    "SegmentationModel",
    "BiRefNetAdapter",
    "ModelManager",
]
