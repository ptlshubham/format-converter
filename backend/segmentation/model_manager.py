"""
Model Manager Singleton
Coordinates model loading, device detection (CPU), ONNX session caching,
and thread-safe inference lifecycle management for the single AI model:
- BiRefNet General FP16 ONNX

RAM-Safe Architecture:
- Lazy model loading (loaded on first request, not at startup)
- Thread-safe single-inference lock (threading.Lock)
- Single-flight model loading protection
- Configurable 90s idle timeout unload (BG_MODEL_IDLE_TIMEOUT)
- Strictly inference-safe unloading (never unloads while inference is active)
"""

import os
import gc
import time
import logging
import threading
import asyncio
from enum import Enum
from typing import Optional, List, Dict, Any

try:
    import onnxruntime as ort
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)
except ImportError:
    ort = None

from .model_adapter import SegmentationModel, BiRefNetAdapter

logger = logging.getLogger("BackgroundRemoval.ModelManager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s [%(levelname)s]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ManagerState(str, Enum):
    IDLE = "IDLE"
    LOADING_MODEL = "LOADING_MODEL"
    READY = "READY"
    PROCESSING = "PROCESSING"
    CLEANING_UP = "CLEANING_UP"
    SHUTTING_DOWN = "SHUTTING_DOWN"


HEAVY_AI_LOCK: threading.Lock = threading.Lock()


class ModelManager:
    """
    Thread-safe Singleton manager for loading, caching, serving, and unloading
    the single BiRefNet General FP16 ONNX model.
    """

    _instance: Optional["ModelManager"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self):
        self.device: str = "CPU"
        self.providers: List[str] = ["CPUExecutionProvider"]
        self.threads: int = 4
        self._detect_device()

        base_models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        self.models_dir = base_models_dir

        self.birefnet_general_path = os.path.join(
            base_models_dir, "segmentation", "birefnet-general", "model_fp16.onnx"
        )

        # Cached adapter / session
        self._birefnet_general: Optional[BiRefNetAdapter] = None
        self._load_times: Dict[str, float] = {}
        self._warmed_up: bool = False

        # State & Concurrency Controls
        self._state: ManagerState = ManagerState.IDLE
        self._state_lock: threading.Lock = threading.Lock()
        self._inference_lock: threading.Lock = HEAVY_AI_LOCK
        self._load_lock: threading.Lock = threading.Lock()

        # Idle Timeout Configuration
        self.idle_timeout_seconds: int = int(os.getenv("BG_MODEL_IDLE_TIMEOUT", "90"))
        self._idle_task: Optional[asyncio.Task] = None
        self._idle_expiry: Optional[float] = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = ModelManager()
            return cls._instance

    @property
    def state(self) -> ManagerState:
        with self._state_lock:
            return self._state

    def set_state(self, new_state: ManagerState) -> None:
        with self._state_lock:
            self._state = new_state

    @property
    def is_loaded(self) -> bool:
        return self._birefnet_general is not None

    @property
    def is_inference_busy(self) -> bool:
        return self._inference_lock.locked()

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def load_times(self) -> Dict[str, float]:
        return self._load_times

    @property
    def idle_timer_active(self) -> bool:
        return self._idle_task is not None and not self._idle_task.done()

    @property
    def idle_remaining_seconds(self) -> Optional[float]:
        if self._idle_expiry is not None:
            rem = self._idle_expiry - time.time()
            return max(0.0, round(rem, 1))
        return None

    def _detect_device(self):
        # Enforce CPU ONLY execution per user instruction (GPU disabled to prevent driver suspension/crashes)
        self.device = "CPU"
        self.providers = ["CPUExecutionProvider"]
        # Cap threads at 4 to avoid exhausting Windows non-paged pool memory
        self.threads = min(4, os.cpu_count() or 4)
        logger.info(f"Configured ONNX Runtime for CPU execution with {self.threads} threads (GPU disabled).")



    def _create_session_options(self) -> Any:
        """
        Creates ONNX Runtime SessionOptions with optimizations for pure CPU execution.
        Memory arena and pattern are enabled so ONNX reuses allocations instead of
        constantly allocating/freeing, reducing peak memory pressure.
        """
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        return opts

    # =========================================================================
    # THREAD-SAFE INFERENCE LOCK
    # =========================================================================

    def try_acquire_inference(self) -> bool:
        """
        Attempts non-blocking acquisition of the single-inference lock.
        Returns True if acquired, False if another inference is currently running.
        Thread-safe for worker thread (asyncio.to_thread) execution.
        """
        acquired = self._inference_lock.acquire(blocking=False)
        if acquired:
            with self._state_lock:
                self._state = ManagerState.PROCESSING
            logger.info("INFERENCE LOCK ACQUIRED: Single-inference ownership granted.")
        else:
            logger.warning("INFERENCE LOCK BUSY: Another inference is already active. Rejecting concurrent request.")
        return acquired

    def release_inference(self) -> None:
        """
        Releases the single-inference lock and restores state to READY or IDLE.
        Thread-safe and guaranteed to never throw.
        """
        try:
            if self._inference_lock.locked():
                self._inference_lock.release()
                logger.info("INFERENCE LOCK RELEASED: Inference completed.")
        except RuntimeError:
            pass

        with self._state_lock:
            if self._birefnet_general is not None:
                self._state = ManagerState.READY
            else:
                self._state = ManagerState.IDLE

    # =========================================================================
    # LAZY SINGLE-FLIGHT MODEL LOADING
    # =========================================================================

    def get_birefnet_general(self, force_reload: bool = False) -> BiRefNetAdapter:
        """
        Returns cached BiRefNet General FP16 adapter.
        Loads the model lazily on first request with single-flight protection (_load_lock).
        Reuses the session for all subsequent requests while warm.
        """
        if self._birefnet_general is not None and not force_reload:
            self._birefnet_reused_this_req = True
            return self._birefnet_general

        with self._load_lock:
            if self._birefnet_general is not None and not force_reload:
                self._birefnet_reused_this_req = True
                return self._birefnet_general

            if not os.path.exists(self.birefnet_general_path):
                raise FileNotFoundError(
                    f"BiRefNet General FP16 model not found at '{self.birefnet_general_path}'. "
                    f"Please ensure 'backend/models/segmentation/birefnet-general/model_fp16.onnx' exists."
                )

            with self._state_lock:
                self._state = ManagerState.LOADING_MODEL

            logger.info(f"MODEL LOAD START: Loading BiRefNet General FP16 (Threads: {self.threads}, Device: {self.device})...")
            t0 = time.perf_counter()
            opts = self._create_session_options()
            session = ort.InferenceSession(
                self.birefnet_general_path, sess_options=opts, providers=self.providers
            )
            self._birefnet_general = BiRefNetAdapter(session, model_variant="general")
            load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self._load_times["birefnet_general"] = load_ms
            self._birefnet_reused_this_req = False
            logger.info(f"MODEL LOAD COMPLETE: BiRefNet General FP16 loaded in {load_ms} ms.")

            with self._state_lock:
                self._state = ManagerState.READY

            return self._birefnet_general

    def load_model(self, force_reload: bool = False) -> BiRefNetAdapter:
        """Explicit load method."""
        return self.get_birefnet_general(force_reload=force_reload)

    def get_model(self, model_name: str = "auto") -> SegmentationModel:
        """Returns the active segmentation adapter."""
        return self.get_birefnet_general()

    # =========================================================================
    # STRICTLY INFERENCE-SAFE 90-SECOND IDLE UNLOAD
    # =========================================================================

    def unload_model(self) -> bool:
        """
        Safely unloads the BiRefNet session and adapter if no inference is running.
        Sequence:
        1. Verify inference lock is NOT held.
        2. Verify state is NOT PROCESSING or CLEANING_UP.
        3. Acquire _load_lock.
        4. Release adapter and ONNX session references.
        5. Invoke gc.collect().
        6. Set state to IDLE.
        Returns True if unloaded, False if inference is active.
        """
        with self._state_lock:
            if self._inference_lock.locked() or self._state in (ManagerState.PROCESSING, ManagerState.CLEANING_UP):
                logger.warning("MODEL UNLOAD ABORTED: Inference is currently active. Model will remain loaded.")
                return False

        with self._load_lock:
            if self._birefnet_general is None:
                with self._state_lock:
                    self._state = ManagerState.IDLE
                return True

            logger.info("MODEL UNLOAD START: Releasing BiRefNet ONNX session after idle period...")
            self._birefnet_general = None
            self._warmed_up = False

            # Explicit garbage collection to release Python object references
            gc.collect()

            with self._state_lock:
                self._state = ManagerState.IDLE

            logger.info("MODEL UNLOAD COMPLETE: BiRefNet session unloaded. State: IDLE")
            return True

    def schedule_idle_unload(self) -> None:
        """
        Schedules a non-blocking asyncio idle unload task after self.idle_timeout_seconds.
        Resets any existing timer.
        """
        self.cancel_idle_timer()

        if self.idle_timeout_seconds <= 0:
            return  # Idle unloading disabled if timeout <= 0

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._idle_expiry = time.time() + self.idle_timeout_seconds

        async def _idle_worker():
            try:
                await asyncio.sleep(self.idle_timeout_seconds)
                logger.info(f"IDLE TIMEOUT: {self.idle_timeout_seconds}s elapsed without requests. Initiating safe unload...")
                self.unload_model()
            except asyncio.CancelledError:
                pass
            finally:
                self._idle_expiry = None

        self._idle_task = loop.create_task(_idle_worker())
        logger.info(f"IDLE TIMER START: Model will unload if inactive for {self.idle_timeout_seconds}s.")

    def cancel_idle_timer(self) -> None:
        """Cancels any pending idle unload task."""
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None
            self._idle_expiry = None
            logger.info("IDLE TIMER CANCELLED: New activity detected. Model remains warm.")

    def shutdown(self) -> None:
        """Gracefully shuts down ModelManager on server stop."""
        with self._state_lock:
            self._state = ManagerState.SHUTTING_DOWN
        self.cancel_idle_timer()
        self.unload_model()
        logger.info("ModelManager shutdown complete.")

