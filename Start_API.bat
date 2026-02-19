@echo off
TITLE IngestForge Engine Bridge (API)
COLOR 0B

echo Starting IngestForge FastAPI Server...
echo API will be available at: http://localhost:8000
echo.

python -m ingestforge.api.main

pause
