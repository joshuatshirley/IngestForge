@echo off
TITLE IngestForge Web Portal (Development)
COLOR 0A

echo ======================================================
echo   IngestForge Web Portal - Development Launcher
echo ======================================================
echo.

REM Start FastAPI Backend in a new window
echo [1/2] Starting Engine Bridge (FastAPI)...
start "IF-API" cmd /c "python -m ingestforge.api.main"

timeout /t 3 /nobreak > nul

REM Start Next.js Frontend
echo [2/2] Starting Web Portal (Next.js)...
cd frontend
npm run dev

pause
