import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Enforce Windows Proactor (IOCP) Event Loop on Windows to prevent WinError 10055 selector buffer exhaustion
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from .api.routes import router
from ..segmentation.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy model loading: model is loaded on the first request rather than during startup
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("FastAPI backend started with lazy model loading (BiRefNet will load on first request).")
    yield
    # Safe shutdown: ensure any loaded model and timers are cleaned up
    try:
        ModelManager.get_instance().shutdown()
    except Exception as e:
        logger.warning(f"Error during backend shutdown: {e}")


app = FastAPI(
    title="Professional AI Background Removal API",
    description="Single-model BiRefNet General FP16 ONNX Background Removal Engine.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

