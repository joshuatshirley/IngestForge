@echo off
REM IngestForge One-Command Installer (Windows)
REM
REM Usage:
REM   install.bat [--no-wizard] [--verbose]
REM
REM JPL Power of Ten Compliance:
REM - Rule #7: All commands check return codes (errorlevel)

setlocal enabledelayedexpansion

echo ========================================================================
echo                   IngestForge Installer (Windows)
echo ========================================================================

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mX Error: Python 3.10+ required but not found.[0m
    echo.
    echo Install Python from: https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [32m✓ Found Python %PYTHON_VERSION%[0m

REM Check Node.js (optional)
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [33m⚠ Warning: Node.js not found ^(frontend features disabled^)[0m
    echo   Install Node.js 18+ from: https://nodejs.org/
) else (
    for /f "tokens=*" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
    echo [32m✓ Found Node.js !NODE_VERSION![0m
)

REM Run Python installer
echo.
echo Running Python installer...
echo.

REM Determine script directory
set SCRIPT_DIR=%~dp0

REM Run with arguments passed through
python "%SCRIPT_DIR%install.py" %*

set INSTALL_EXIT_CODE=%errorlevel%

if %INSTALL_EXIT_CODE% equ 0 (
    echo.
    echo ========================================================================
    echo [32m✓ Installation successful![0m
    echo ========================================================================
    echo.
    echo Next steps:
    echo   1. Activate virtual environment:
    echo      venv\Scripts\activate.bat
    echo.
    echo   2. Run IngestForge (convenience script):
    echo      .\IngestForge.bat --help
    echo.
    echo   3. Or run setup wizard:
    echo      python -m ingestforge.cli.setup_wizard
    echo.
) else (
    echo.
    echo ========================================================================
    echo [31m✗ Installation failed ^(exit code: %INSTALL_EXIT_CODE%^)[0m
    echo ========================================================================
    echo.
    echo Troubleshooting:
    echo   1. Ensure Python 3.10+ is installed
    echo   2. Run as Administrator if permission errors occur
    echo   3. Try with --verbose flag for more details
    echo   4. See docs: https://docs.ingestforge.io/troubleshooting
    echo.
)

pause
exit /b %INSTALL_EXIT_CODE%
