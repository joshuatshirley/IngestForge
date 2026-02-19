@echo off
setlocal enabledelayedexpansion
title IngestForge Mission-Critical CLI
cd /d "%~dp0"

REM Version Info
set "IF_VERSION=v1.1.0-stable"

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ not found.
    echo Please install Python from https://python.org and add it to your PATH.
    pause
    exit /b 1
)

REM Ensure IngestForge is in editable mode for development
if not exist "ingestforge.egg-info" (
    echo [SETUP] Initializing IngestForge environment...
    pip install -e . --quiet
)

REM Check for command line arguments
if "%1"=="--cli" goto :cli_splash
if "%1"=="-c" goto :cli_splash

REM Default: Launch interactive TUI menu
echo [LAUNCH] Starting Interactive Research Menu...
python -m ingestforge.cli.interactive.menu
if errorlevel 1 (
    echo [ERROR] Interactive menu failed. Falling back to command splash.
    goto :cli_splash
)
exit /b 0

:cli_splash
REM Launch PowerShell with IngestForge CLI Branding
powershell -NoExit -Command "& {
    $host.UI.RawUI.WindowTitle = 'IngestForge CLI - !IF_VERSION!'
    Write-Host ''
    Write-Host '  ___                       _   _____                      ' -ForegroundColor Cyan
    Write-Host ' |_ _|_ __   __ _  ___  ___| |_|  ___|__  _ __ __ _  ___  ' -ForegroundColor Cyan
    Write-Host '  | || ''_ \\ / _` |/ _ \\/ __| __| |_ / _ \\| ''__/ _` |/ _ \\ ' -ForegroundColor Cyan
    Write-Host '  | || | | | (_| |  __/\\__ \\ |_|  _| (_) | | | (_| |  __/ ' -ForegroundColor Cyan
    Write-Host ' |___|_| |_|\\__, |\\___||___/\\__|_|  \\___/|_|  \\__, |\\___| ' -ForegroundColor Cyan
    Write-Host '            |___/                             |___/  !IF_VERSION! ' -ForegroundColor Cyan
    Write-Host ''
    Write-Host ' > MISSION-CRITICAL DOCUMENT INTELLIGENCE FRAMEWORK' -ForegroundColor Gray
    Write-Host ' > JPL POWER OF TEN COMPLIANT' -ForegroundColor Green
    Write-Host ''
    Write-Host ' [QUICK START]' -ForegroundColor Yellow
    Write-Host '   ingestforge init <name>    Initialize workspace'
    Write-Host '   ingestforge ingest <dir>   Process documents'
    Write-Host '   ingestforge query \"...\"    Search knowledge base'
    Write-Host '   ingestforge status         Check engine health'
    Write-Host ''
    Write-Host ' Type ''ingestforge --help'' for the full command reference.' -ForegroundColor Green
    Write-Host ''
}"
