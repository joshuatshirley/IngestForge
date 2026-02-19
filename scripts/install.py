#!/usr/bin/env python3
"""IngestForge One-Command Installer.

One-Command-Install
Epic EP-31 (MVP Readiness)
Status: IN_PROGRESS (2026-02-18T16:30:00Z)

Epic AC Implementation Map:
- Cross-platform scripts → install.py (this file)
- Dependency detection → check_python_version(), check_node_version(), check_system_resources()
- Automated installation → create_virtualenv(), install_dependencies(), build_frontend()
- Setup wizard integration → launch_setup_wizard(), --no-wizard flag
- Error handling & rollback → InstallationError, rollback_installation()

JPL Power of Ten Compliance:
- Rule #2: All loops bounded with MAX constants
- Rule #4: All functions <60 lines
- Rule #7: All subprocess calls check return values
- Rule #9: 100% type hints

Usage:
    python scripts/install.py                    # Full installation with wizard
    python scripts/install.py --no-wizard        # Skip setup wizard
    python scripts/install.py --skip-frontend    # Skip frontend build (development)
    python scripts/install.py --verbose          # Detailed output
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Constants (JPL Rule #2: Fixed bounds)
# =============================================================================

REQUIRED_PYTHON_VERSION = (3, 10)
REQUIRED_NODE_VERSION = (18, 0)
MIN_RAM_GB = 2.0
MIN_DISK_GB = 5.0
MAX_INSTALL_RETRIES = 3
INSTALL_TIMEOUT_SECONDS = 600  # 10 minutes max per operation
WIZARD_TIMEOUT_SECONDS = 300  # 5 minutes max for wizard

# JPL Rule #2: Bounded iterations
MAX_DEPENDENCY_CHECKS = 10
MAX_SEARCH_DEPTH = 5


# =============================================================================
# Exceptions (Epic )
# =============================================================================


class InstallationError(Exception):
    """Base exception for installation errors."""

    pass


class PrerequisiteError(InstallationError):
    """Raised when prerequisites are not met ()."""

    pass


class DependencyInstallError(InstallationError):
    """Raised when dependency installation fails ()."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class InstallMetadata:
    """Installation metadata.

    Epic Stores install metadata (version, date, platform).
    JPL Rule #9: 100% type hints on all attributes.
    """

    version: str
    platform: str
    python_version: str
    node_version: str
    install_date: str
    install_path: str
    venv_path: str
    frontend_built: bool = False
    wizard_completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "version": self.version,
            "platform": self.platform,
            "python_version": self.python_version,
            "node_version": self.node_version,
            "install_date": self.install_date,
            "install_path": self.install_path,
            "venv_path": self.venv_path,
            "frontend_built": self.frontend_built,
            "wizard_completed": self.wizard_completed,
        }


@dataclass
class SystemInfo:
    """System resource information.

    Epic Validates system requirements (2GB+ RAM, 5GB+ disk).
    """

    ram_gb: float
    disk_gb: float
    platform_name: str
    meets_requirements: bool = True
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Prerequisite Checks (Epic )
# =============================================================================


def check_python_version() -> Tuple[bool, str, str]:
    """Check if Python version meets requirements.

    Epic Auto-detects Python version (requires 3.10+)

    JPL Compliance:
    - Rule #7: Returns explicit success status
    - Rule #9: Complete type hints

    Returns:
        (success: bool, message: str, version_string: str)

    Tests: test_check_python_version
    """
    current = sys.version_info[:2]
    required = REQUIRED_PYTHON_VERSION
    version_str = f"{current[0]}.{current[1]}"

    if current >= required:
        return (True, f"✓ Python {version_str}", version_str)

    msg = (
        f"✗ Python {required[0]}.{required[1]}+ required (found {version_str})\n"
        f"  Install: https://www.python.org/downloads/\n"
        f"  Docs: https://docs.ingestforge.io/install#python"
    )
    return (False, msg, version_str)


def check_node_version() -> Tuple[bool, str, str]:
    """Check if Node.js version meets requirements.

    Epic Auto-detects Node.js version (requires 18+)

    JPL Compliance:
    - Rule #4: <60 lines
    - Rule #7: Checks subprocess return value

    Returns:
        (success: bool, message: str, version_string: str)

    Tests: test_check_node_version
    """
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,  # JPL Rule #7: Timeout protection
        )

        if result.returncode != 0:
            msg = (
                "✗ Node.js not found\n"
                "  Install: https://nodejs.org/\n"
                "  Docs: https://docs.ingestforge.io/install#nodejs"
            )
            return (False, msg, "")

        # Parse version: "v18.12.0" -> (18, 12)
        version_str = result.stdout.strip().lstrip("v")
        parts = version_str.split(".")
        if len(parts) < 2:
            return (False, "✗ Could not parse Node.js version", "")

        major = int(parts[0])
        minor = int(parts[1])

        required = REQUIRED_NODE_VERSION
        if (major, minor) >= required:
            return (True, f"✓ Node.js {major}.{minor}", version_str)

        msg = (
            f"✗ Node.js {required[0]}+ required (found {major}.{minor})\n"
            f"  Install: https://nodejs.org/\n"
            f"  Docs: https://docs.ingestforge.io/install#nodejs"
        )
        return (False, msg, version_str)

    except subprocess.TimeoutExpired:
        return (False, "✗ Node.js check timed out", "")
    except (ValueError, IndexError):
        return (False, "✗ Could not parse Node.js version", "")
    except FileNotFoundError:
        msg = "✗ Node.js not found\n" "  Install: https://nodejs.org/"
        return (False, msg, "")
    except Exception as e:
        return (False, f"✗ Node.js check failed: {e}", "")


def get_total_ram() -> float:
    """Get total system RAM in GB (cross-platform).

    Cross-platform support (Windows, macOS, Linux).
    Detects system RAM.

    JPL Rule #4: <60 lines
    JPL Rule #9: 100% type hints

    Returns:
        Total RAM in GB, or 0.0 if could not determine.
    """
    sys_platform = platform.system()
    try:
        if sys_platform == "Windows":
            return _get_ram_windows()
        if sys_platform == "Darwin":
            return _get_ram_macos()
        if sys_platform == "Linux":
            return _get_ram_linux()
    except Exception:
        pass
    return 0.0


def _get_ram_windows() -> float:
    """Get Windows RAM using ctypes.

    Windows support via ctypes.
    """
    import ctypes

    kernel32 = ctypes.windll.kernel32
    c_ulong = ctypes.c_ulong

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", c_ulong),
            ("dwMemoryLoad", c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return stat.ullTotalPhys / (1024**3)
    return 0.0


def _get_ram_macos() -> float:
    """Get macOS RAM using sysctl.

    macOS support via sysctl.
    """
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        return int(result.stdout.strip()) / (1024**3)
    return 0.0


def _get_ram_linux() -> float:
    """Get Linux RAM from /proc/meminfo.

    Linux support via /proc/meminfo.
    JPL Rule #2: Bounded loops.
    """
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return 0.0

    # JPL Rule #2: Explicit loop bound
    max_lines = 1000
    try:
        with open(meminfo_path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if "MemTotal" in line:
                    return int(line.split()[1]) / (1024**2)
    except (IOError, ValueError, IndexError):
        pass
    return 0.0


def check_system_resources() -> SystemInfo:
    """Check system resources meet requirements.

    Detects total system RAM.
    Detects free disk space.
    Cross-platform implementation.

    JPL Rule #4: <60 lines

    Returns:
        SystemInfo with resource availability and warnings

    Tests: test_check_system_resources
    """
    warnings: List[str] = []
    ram_gb = get_total_ram()
    disk_gb = 0.0

    # Get disk space
    try:
        disk_info = shutil.disk_usage(Path.cwd())
        disk_gb = disk_info.free / (1024**3)
    except Exception:
        warnings.append("⚠️  Could not determine disk space - assuming sufficient")

    if ram_gb == 0.0:
        warnings.append("⚠️  Could not determine RAM - assuming sufficient")

    # Check requirements
    meets_requirements = True

    if 0.0 < ram_gb < MIN_RAM_GB:
        warnings.append(f"⚠️  Low RAM: {ram_gb:.1f}GB (recommended: {MIN_RAM_GB}GB+)")
        meets_requirements = False

    if 0.0 < disk_gb < MIN_DISK_GB:
        warnings.append(
            f"⚠️  Low disk: {disk_gb:.1f}GB free (recommended: {MIN_DISK_GB}GB+)"
        )
        meets_requirements = False

    return SystemInfo(
        ram_gb=ram_gb,
        disk_gb=disk_gb,
        platform_name=platform.system(),
        meets_requirements=meets_requirements,
        warnings=warnings,
    )


# =============================================================================
# Installation Functions (Epic )
# =============================================================================


def create_virtualenv(
    project_root: Path, verbose: bool = False
) -> Tuple[bool, Optional[Path], str]:
    """Create Python virtual environment.

    Epic Creates Python virtual environment automatically

    JPL Compliance:
    - Rule #2: No unbounded loops
    - Rule #7: Checks all subprocess calls
    - Rule #4: <60 lines

    Args:
        project_root: Root directory for IngestForge
        verbose: Print detailed output

    Returns:
        (success: bool, venv_path: Optional[Path], message: str)

    Tests: test_create_virtualenv
    """
    venv_path = project_root / "venv"

    # Remove existing venv if present
    if venv_path.exists():
        if verbose:
            print(f"  Removing old venv: {venv_path}")
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            return (False, None, f"Failed to remove old venv: {e}")

    if verbose:
        print(f"  Creating venv: {venv_path}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return (False, None, f"venv creation failed: {result.stderr}")

        return (True, venv_path, "✓ Virtual environment created")

    except subprocess.TimeoutExpired:
        return (False, None, "venv creation timed out")
    except Exception as e:
        return (False, None, f"venv creation failed: {e}")


def install_python_dependencies(
    venv_path: Path, project_root: Path, verbose: bool = False
) -> Tuple[bool, str]:
    """Install Python packages from requirements.txt.

    Epic Installs Python dependencies from requirements.txt

    JPL Compliance:
    - Rule #2: Bounded retry loop (MAX_INSTALL_RETRIES)
    - Rule #4: <60 lines

    Args:
        venv_path: Path to virtual environment
        project_root: Project root with requirements.txt
        verbose: Print detailed output

    Returns:
        (success: bool, message: str)

    Tests: test_install_python_dependencies
    """
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        return (False, f"requirements.txt not found: {requirements_file}")

    # Determine pip executable
    if sys.platform == "win32":
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        pip_exe = venv_path / "bin" / "pip"

    if not pip_exe.exists():
        return (False, f"pip not found in venv: {pip_exe}")

    # JPL Rule #2: Bounded retry loop
    for attempt in range(MAX_INSTALL_RETRIES):
        if verbose or attempt > 0:
            print(f"  Attempt {attempt + 1}/{MAX_INSTALL_RETRIES}...")

        try:
            result = subprocess.run(
                [str(pip_exe), "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT_SECONDS,
            )

            if result.returncode == 0:
                return (True, "✓ Python dependencies installed")

            if verbose:
                print(f"  Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            if attempt < MAX_INSTALL_RETRIES - 1:
                continue

    return (False, f"pip install failed after {MAX_INSTALL_RETRIES} attempts")


def install_node_dependencies(
    project_root: Path, verbose: bool = False
) -> Tuple[bool, str]:
    """Install Node.js dependencies for frontend.

    Epic Installs Node.js dependencies (frontend)

    JPL Rule #7: Checks subprocess return value

    Args:
        project_root: Project root with frontend/package.json
        verbose: Print detailed output

    Returns:
        (success: bool, message: str)

    Tests: test_install_node_dependencies
    """
    frontend_dir = project_root / "frontend"
    package_json = frontend_dir / "package.json"

    if not package_json.exists():
        return (False, f"package.json not found: {package_json}")

    if verbose:
        print(f"  Running npm install in {frontend_dir}")

    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
            shell=(sys.platform == "win32"),
        )

        if result.returncode != 0:
            return (False, f"npm install failed: {result.stderr[:200]}")

        return (True, "✓ Node.js dependencies installed")

    except subprocess.TimeoutExpired:
        return (False, "npm install timed out")
    except Exception as e:
        return (False, f"npm install failed: {e}")


def build_frontend(project_root: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Build frontend static assets.

    Epic Builds frontend static assets

    JPL Rule #7: Checks subprocess return value

    Args:
        project_root: Project root with frontend/
        verbose: Print detailed output

    Returns:
        (success: bool, message: str)

    Tests: test_build_frontend
    """
    frontend_dir = project_root / "frontend"

    if verbose:
        print(f"  Running npm build in {frontend_dir}")

    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
            shell=(sys.platform == "win32"),
        )

        if result.returncode != 0:
            return (False, f"Frontend build failed: {result.stderr[:200]}")

        # Verify output
        output_dir = frontend_dir / "out"
        if not output_dir.exists():
            return (False, "Frontend built but 'out' directory not found")

        return (True, "✓ Frontend built successfully")

    except subprocess.TimeoutExpired:
        return (False, "Frontend build timed out")
    except Exception as e:
        return (False, f"Frontend build failed: {e}")


def create_config_directory() -> Tuple[bool, Path, str]:
    """Create ~/.ingestforge/ config directory.

    Epic Creates ~/.ingestforge/ config directory

    Returns:
        (success: bool, config_path: Path, message: str)

    Tests: test_create_config_directory
    """
    config_dir = Path.home() / ".ingestforge"

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "data").mkdir(exist_ok=True)
        (config_dir / "logs").mkdir(exist_ok=True)

        return (True, config_dir, f"✓ Config directory: {config_dir}")

    except Exception as e:
        return (False, config_dir, f"Failed to create config: {e}")


def save_install_metadata(
    config_dir: Path, metadata: InstallMetadata
) -> Tuple[bool, str]:
    """Save installation metadata to file.

    Epic Stores install metadata (version, date, platform)

    Args:
        config_dir: Config directory path
        metadata: Installation metadata

    Returns:
        (success: bool, message: str)

    Tests: test_save_install_metadata
    """
    metadata_file = config_dir / "install_metadata.json"

    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return (True, f"✓ Metadata saved: {metadata_file}")

    except Exception as e:
        return (False, f"Failed to save metadata: {e}")


# =============================================================================
# Setup Wizard Integration (Epic )
# =============================================================================


def launch_setup_wizard(venv_path: Path) -> Tuple[bool, str]:
    """Launch setup wizard after installation.

    Epic Launches setup wizard automatically after install

    Args:
        venv_path: Path to virtual environment

    Returns:
        (success: bool, message: str)

    Tests: test_launch_setup_wizard
    """
    # Determine python executable in venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if not python_exe.exists():
        return (False, f"Python not found in venv: {python_exe}")

    print("\n" + "=" * 60)
    print("Launching Setup Wizard...")
    print("=" * 60 + "\n")

    try:
        result = subprocess.run(
            [str(python_exe), "-m", "ingestforge.cli.setup_wizard"],
            timeout=WIZARD_TIMEOUT_SECONDS,
        )

        if result.returncode == 0:
            return (True, "✓ Setup wizard completed")
        else:
            return (False, "Setup wizard exited with errors")

    except subprocess.TimeoutExpired:
        return (False, "Setup wizard timed out")
    except KeyboardInterrupt:
        return (False, "Setup wizard cancelled by user")
    except Exception as e:
        return (False, f"Setup wizard failed: {e}")


# =============================================================================
# Rollback (Epic )
# =============================================================================


def rollback_installation(
    venv_path: Optional[Path], config_dir: Optional[Path]
) -> None:
    """Rollback partial installation on failure.

    Epic Rollback on partial install failure

    JPL Rule #2: Bounded cleanup operations

    Args:
        venv_path: Virtual environment to remove
        config_dir: Config directory to remove

    Tests: test_rollback_installation
    """
    print("\n⚠️  Rolling back installation...")

    if venv_path and venv_path.exists():
        try:
            print(f"  Removing {venv_path}...")
            shutil.rmtree(venv_path)
        except Exception as e:
            print(f"  Warning: Could not remove venv: {e}")

    if config_dir and config_dir.exists():
        try:
            print(f"  Removing {config_dir}...")
            shutil.rmtree(config_dir)
        except Exception as e:
            print(f"  Warning: Could not remove config: {e}")

    print("✓ Rollback complete")


# =============================================================================
# Main Installation Orchestrator
# =============================================================================


def _verify_prerequisites(parsed: argparse.Namespace) -> Tuple[bool, str, str]:
    """Verify all software and hardware prerequisites.

    Exits/warns if requirements not met.
    Respects --skip-hw-check override.
    QA-0101: Validates pip presence for clean OS.

    JPL Rule #4: <60 lines.
    """
    print("\n[1/6] Checking Prerequisites...")
    py_ok, py_msg, py_ver = check_python_version()
    node_ok, node_msg, node_ver = check_node_version()
    sys_info = check_system_resources()

    # Check for pip
    pip_ok = True
    try:
        import pip
    except ImportError:
        pip_ok = False

    print(f"  {py_msg}")
    print(f"  {node_msg}")
    if not pip_ok:
        print("  ✗ pip not found (required for installation)")

    for warning in sys_info.warnings:
        print(f"  {warning}")

    if not (py_ok and node_ok and pip_ok):
        print("\n✗ Required software prerequisites not met (Python/Node/pip)")
        return False, "", ""

    if not sys_info.meets_requirements and not parsed.skip_hw_check:
        print("\n✗ System requirements not met: 2GB+ RAM, 5GB+ disk space required")
        print("  Use --skip-hw-check to override (not recommended)")
        return False, "", ""

    if not sys_info.meets_requirements and parsed.skip_hw_check:
        print(
            "\n⚠️  System requirements not met, but --skip-hw-check enabled. Proceeding..."
        )

    return True, py_ver, node_ver


def _perform_installation(
    project_root: Path, parsed: argparse.Namespace
) -> Tuple[Optional[Path], Optional[Path]]:
    """Execute the core installation steps.

    JPL Rule #4: <60 lines.
    """
    # Phase 2: Create venv
    print("\n[2/6] Creating Virtual Environment...")
    success, venv_path, msg = create_virtualenv(project_root, parsed.verbose)
    print(f"  {msg}")
    if not success or venv_path is None:
        return None, None

    # Phase 3: Install Python deps
    print("\n[3/6] Installing Python Dependencies...")
    success, msg = install_python_dependencies(venv_path, project_root, parsed.verbose)
    print(f"  {msg}")
    if not success:
        rollback_installation(venv_path, None)
        return None, None

    # Phase 4: Frontend
    if not parsed.skip_frontend:
        if not _handle_frontend_install(project_root, venv_path, parsed.verbose):
            return None, None
    else:
        print("\n[4-5/6] Skipping frontend (--skip-frontend)")

    # Phase 5: Config
    print("\n[6/6] Setting Up Configuration...")
    success, config_dir, msg = create_config_directory()
    print(f"  {msg}")
    if not success:
        rollback_installation(venv_path, config_dir)
        return None, None

    return venv_path, config_dir


def _handle_frontend_install(
    project_root: Path, venv_path: Path, verbose: bool
) -> bool:
    """Helper for frontend steps to keep _perform_installation short."""
    print("\n[4/6] Installing Node.js Dependencies...")
    success, msg = install_node_dependencies(project_root, verbose)
    print(f"  {msg}")
    if not success:
        rollback_installation(venv_path, None)
        return False

    print("\n[5/6] Building Frontend...")
    success, msg = build_frontend(project_root, verbose)
    print(f"  {msg}")
    if not success:
        rollback_installation(venv_path, None)
        return False
    return True


def _finalize_installation(
    project_root: Path,
    venv_path: Path,
    config_dir: Path,
    py_ver: str,
    node_ver: str,
    parsed: argparse.Namespace,
) -> None:
    """Save metadata and optionally launch wizard.

    JPL Rule #4: <60 lines.
    """
    metadata = InstallMetadata(
        version="1.0.0",
        platform=platform.system(),
        python_version=py_ver,
        node_version=node_ver,
        install_date=datetime.now().isoformat(),
        install_path=str(project_root),
        venv_path=str(venv_path),
        frontend_built=not parsed.skip_frontend,
        wizard_completed=False,
    )
    save_install_metadata(config_dir, metadata)

    print("\n" + "=" * 60)
    print("✓ Installation Complete!")
    print("=" * 60)

    if not parsed.no_wizard:
        wizard_ok, _ = launch_setup_wizard(venv_path)
        if wizard_ok:
            metadata.wizard_completed = True
            save_install_metadata(config_dir, metadata)
    else:
        print("\nRun setup wizard later: ingestforge setup")

    _print_next_steps(venv_path)


def _print_next_steps(venv_path: Path) -> None:
    """Print post-install instructions."""
    print("\nNext Steps:")
    if sys.platform == "win32":
        print(f"  Activate venv: {venv_path}\\Scripts\\activate")
    else:
        print(f"  Activate venv: source {venv_path}/bin/activate")
    print("  Run IngestForge: ingestforge --help")
    print("=" * 60)


def main(args: Optional[List[str]] = None) -> int:
    """Main installation entry point.

    JPL Rule #4: <60 lines (coordinator).
    """
    parser = argparse.ArgumentParser(description="IngestForge One-Command Installer")
    parser.add_argument("--no-wizard", action="store_true")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-hw-check", action="store_true")
    parsed = parser.parse_args(args)

    print("=" * 60 + "\nIngestForge Installation\n" + "=" * 60)
    project_root = Path(__file__).parent.parent

    try:
        ok, py_ver, node_ver = _verify_prerequisites(parsed)
        if not ok:
            return 1

        venv_path, config_dir = _perform_installation(project_root, parsed)
        if not venv_path or not config_dir:
            return 1

        _finalize_installation(
            project_root, venv_path, config_dir, py_ver, node_ver, parsed
        )
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled")
        return 1
    except Exception as e:
        print(f"\n\n✗ Installation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
