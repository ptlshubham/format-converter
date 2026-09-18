"""
Top-level entry point for FastAPI backend
Allows running: uvicorn backend.main:app
"""

from .background_remover.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
