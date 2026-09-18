@echo off
title Image Studio - All In One Starter
color 0b
echo ===================================================
echo     STARTING IMAGE CONVERTER + BACKGROUND REMOVER
echo ===================================================
echo.
cd /d "%~dp0"

echo [1/2] Starting Python Backend on Port 8000...
start "Background Remover Backend" cmd /k "py -3.13 -m uvicorn backend.background_remover.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting Angular Frontend on Port 4200...
start "Angular Frontend" cmd /k "npm start"

echo.
echo ===================================================
echo  ALL SERVICES STARTED!
echo  Backend:  http://127.0.0.1:8000
echo  Frontend: http://localhost:4200
echo ===================================================
timeout /t 4 >nul
start http://localhost:4200
exit
