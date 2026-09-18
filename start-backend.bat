@echo off
title Background Remover Backend
echo Starting Background Remover Backend on http://127.0.0.1:8000...
cd /d "%~dp0"
py -3.13 -m uvicorn backend.background_remover.main:app --host 127.0.0.1 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    python -m uvicorn backend.background_remover.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
