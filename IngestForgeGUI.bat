@echo off
setlocal
title IngestForge Foundry GUI Launcher
cd /d "%~dp0"

echo ======================================================
echo   INGESTFORGE FOUNDRY - MISSION-CRITICAL WORKBENCH
echo ======================================================
echo.

REM 1. Backend Check & Start
echo [1/3] Checking Engine Bridge (FastAPI)...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing backend dependencies...
    pip install -e .[all] --quiet
)

echo [2/3] Starting Engine Bridge...
start "IF-API-ENGINE" cmd /c "python -m ingestforge.api.main"

REM Wait for backend to spin up
timeout /t 4 /nobreak > nul

REM 2. Frontend Check & Start
echo [3/3] Starting Web Portal (Next.js)...
if not exist "frontend\node_modules" (
    echo [SETUP] Installing frontend dependencies...
    cd frontend && npm install && cd ..
)

cd frontend
npm run dev

echo.
echo [READY] Foundry UI should be live at http://localhost:3000
echo Close this window to terminate both services.
pause
