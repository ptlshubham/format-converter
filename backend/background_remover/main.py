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

from .settings import settings
from .guards import RequestGuards
from .api.routes import router, dev_router
from ..segmentation.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy model loading: model is loaded on the first request rather than during startup
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("FastAPI backend started with lazy model loading (BiRefNet will load on first request).")
    logger.info(
        f"Image AI settings: env={settings.env}, origins={','.join(settings.allowed_origins)}, "
        f"rate_limit={settings.rate_limit_per_minute}/min, "
        f"max_upload={settings.max_upload_bytes // (1024 * 1024)}MB, max_json={settings.max_json_bytes // (1024 * 1024)}MB, "
        f"profiler_log={'on' if settings.profiler_log else 'off'}"
    )
    yield
    # Safe shutdown: ensure any loaded model and timers are cleaned up
    try:
        ModelManager.get_instance().shutdown()
    except Exception as e:
        logger.warning(f"Error during backend shutdown: {e}")


# Production hides the interactive docs and the schema along with the dev routes.
docs_enabled = not settings.is_production

app = FastAPI(
    title="Professional AI Background Removal API",
    description="Single-model BiRefNet General FP16 ONNX Background Removal Engine.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
)

# Middleware added last runs first: CORS wraps the guards, so a 429 or 413 from
# the guards still carries CORS headers and the page can read it.
app.add_middleware(RequestGuards, settings=settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,  # the Kardlyz page sends no cookies or auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(router)
if not settings.is_production:
    app.include_router(dev_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
