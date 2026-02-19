#!/usr/bin/env python3
"""Platform installer packaging for IngestForge.

Creates platform-specific installers (EXE, DMG, DEB, AppImage) from
standalone binaries with desktop shortcuts, icons, and file associations.

Platform Installers
Epic: EP-31 (MVP Readiness)

JPL Power of Ten Compliance:
- Rule #2: Bounded loops (max_levels, MAX_ITERATIONS)
- Rule #4: All functions < 60 lines (Refactored)
- Rule #7: Check all return values (subprocess.run, Path.exists)
- Rule #9: 100% type hints
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Constants (Rule #2: Fixed bounds)
MAX_PACKAGE_SIZE_MB = 500
PACKAGE_TIMEOUT_SECONDS = 1800  # 30 minutes
MAX_DIR_DEPTH = 20


class InstallerFormat(str, Enum):
    """Available installer formats."""

    MSI = "msi"
    DMG = "dmg"
    DEB = "deb"
    RPM = "rpm"
    APPIMAGE = "appimage"
    NSIS = "nsis"


@dataclass
class PackageMetadata:
    """Metadata for the package."""

    name: str = "IngestForge"
    version: str = "1.0.0"
    description: str = "AI-powered knowledge management system"
    author: str = "IngestForge Team"
    license: str = "MIT"
    homepage: str = "https://github.com/ingestforge/ingestforge"
    identifier: str = "com.ingestforge.app"


@dataclass
class InstallerConfig:
    """Configuration for installer creation."""

    format: InstallerFormat
    binary_path: Path
    output_dir: Path = field(default_factory=lambda: Path("dist/installers"))
    metadata: PackageMetadata = field(default_factory=PackageMetadata)
    icon: Optional[Path] = None
    create_shortcut: bool = True
    add_to_path: bool = True


@dataclass
class InstallerResult:
    """Result of installer creation."""

    success: bool
    format: InstallerFormat
    output_path: Optional[Path] = None
    error: Optional[str] = None
    size_bytes: int = 0


class InstallerPackager:
    """Creates platform-specific installers."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or self._find_project_root()

    def _find_project_root(self) -> Path:
        """JPL Rule #2: Bounded levels."""
        current = Path(__file__).resolve().parent
        for _ in range(MAX_DIR_DEPTH):
            if (current / "pyproject.toml").exists():
                return current
            if current == current.parent:
                break
            current = current.parent
        return Path.cwd()

    def check_prerequisites(self, config: InstallerConfig) -> List[str]:
        """Verify build tools are available."""
        missing = []
        if not config.binary_path.exists():
            missing.append(f"Binary not found: {config.binary_path}")

        tools = {
            InstallerFormat.DMG: ["hdiutil"],
            InstallerFormat.DEB: ["dpkg-deb"],
            InstallerFormat.NSIS: ["makensis"],
            InstallerFormat.APPIMAGE: ["appimagetool"],
        }

        for tool in tools.get(config.format, []):
            if not shutil.which(tool):
                missing.append(f"{tool} not found")
        return missing

    def package(self, config: InstallerConfig) -> InstallerResult:
        """Create installer package."""
        missing = self.check_prerequisites(config)
        if missing:
            return InstallerResult(
                False, config.format, error=f"Missing: {', '.join(missing)}"
            )

        config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            handlers = {
                InstallerFormat.DMG: self._create_dmg,
                InstallerFormat.DEB: self._create_deb,
                InstallerFormat.NSIS: self._create_nsis,
                InstallerFormat.APPIMAGE: self._create_appimage,
            }
            handler = handlers.get(config.format)
            if not handler:
                return InstallerResult(
                    False, config.format, error=f"Unsupported: {config.format}"
                )
            return handler(config)
        except Exception as e:
            return InstallerResult(False, config.format, error=str(e))

    def _create_nsis(self, config: InstallerConfig) -> InstallerResult:
        """Create Windows EXE installer via NSIS."""
        nsi_content = self._generate_nsis_script(config)
        nsi_path = config.output_dir / "installer.nsi"
        nsi_path.write_text(nsi_content)

        result = subprocess.run(
            ["makensis", str(nsi_path)],
            capture_output=True,
            text=True,
            timeout=PACKAGE_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            return InstallerResult(
                False, config.format, error=f"NSIS failed: {result.stderr}"
            )

        output_path = config.output_dir / f"{config.metadata.name}-Setup.exe"
        return InstallerResult(
            True, config.format, output_path, size_bytes=output_path.stat().st_size
        )

    def _generate_nsis_script(self, config: InstallerConfig) -> str:
        """Generate NSIS script with shortcuts and associations."""
        return f"""
!include "MUI2.nsh"
!define PRODUCT_NAME "{config.metadata.name}"
!define PRODUCT_VERSION "{config.metadata.version}"
Name "${{PRODUCT_NAME}} ${{PRODUCT_VERSION}}"
OutFile "{config.metadata.name}-Setup.exe"
InstallDir "$PROGRAMFILES64\\${{PRODUCT_NAME}}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath $INSTDIR
    File "{config.binary_path}"
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    CreateShortcut "$DESKTOP\\${{PRODUCT_NAME}}.lnk" "$INSTDIR\\{config.binary_path.name}"
    
    # File Association
    WriteRegStr HKCR ".ingestforge" "" "${{PRODUCT_NAME}}Project"
    WriteRegStr HKCR "${{PRODUCT_NAME}}Project\\shell\\open\\command" "" '"$INSTDIR\\{config.binary_path.name}" "%1"'
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\uninstall.exe"
    Delete "$INSTDIR\\{config.binary_path.name}"
    RMDir /r "$INSTDIR"
    Delete "$DESKTOP\\${{PRODUCT_NAME}}.lnk"
SectionEnd
"""

    def _create_dmg(self, config: InstallerConfig) -> InstallerResult:
        """Create macOS DMG with code signing."""
        output_name = f"{config.metadata.name}-{config.metadata.version}.dmg"
        output_path = config.output_dir / output_name

        with tempfile.TemporaryDirectory() as tmpdir:
            app_path = Path(tmpdir) / f"{config.metadata.name}.app"
            macos_dir = app_path / "Contents" / "MacOS"
            macos_dir.mkdir(parents=True)
            shutil.copy2(config.binary_path, macos_dir / config.metadata.name)
            (macos_dir / config.metadata.name).chmod(0o755)

            # Code Signing
            identity = os.getenv("MACOS_CERT_IDENTITY")
            if identity:
                subprocess.run(["codesign", "-s", identity, str(app_path)], check=True)

            subprocess.run(
                [
                    "hdiutil",
                    "create",
                    "-volname",
                    config.metadata.name,
                    "-srcfolder",
                    tmpdir,
                    "-ov",
                    "-format",
                    "UDZO",
                    str(output_path),
                ],
                check=True,
            )

        return InstallerResult(
            True, config.format, output_path, size_bytes=output_path.stat().st_size
        )

    def _create_deb(self, config: InstallerConfig) -> InstallerResult:
        """Create Debian package. Refactored for JPL Rule #4."""
        output_name = (
            f"{config.metadata.name.lower()}_{config.metadata.version}_amd64.deb"
        )
        output_path = config.output_dir / output_name

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_root = Path(tmpdir)
            self._setup_deb_filesystem(pkg_root, config)
            self._write_deb_metadata(pkg_root, config)
            subprocess.run(
                ["dpkg-deb", "--build", str(pkg_root), str(output_path)], check=True
            )

        return InstallerResult(
            True, config.format, output_path, size_bytes=output_path.stat().st_size
        )

    def _setup_deb_filesystem(self, root: Path, config: InstallerConfig) -> None:
        """Create directory structure and copy binary."""
        (root / "usr/bin").mkdir(parents=True)
        (root / "usr/share/applications").mkdir(parents=True)
        shutil.copy2(
            config.binary_path, root / "usr/bin" / config.metadata.name.lower()
        )

    def _write_deb_metadata(self, root: Path, config: InstallerConfig) -> None:
        """Write desktop file and control metadata."""
        name = config.metadata.name.lower()
        desktop = root / "usr/share/applications" / f"{name}.desktop"
        desktop.write_text(
            f"[Desktop Entry]\nName={config.metadata.name}\nExec={name} %f\nType=Application\nMimeType=application/x-ingestforge;\n"
        )

        debian_dir = root / "DEBIAN"
        debian_dir.mkdir()
        (debian_dir / "control").write_text(
            f"Package: {name}\nVersion: {config.metadata.version}\nSection: utils\nArchitecture: amd64\nMaintainer: {config.metadata.author}\nDescription: {config.metadata.description}\n"
        )

    def _create_appimage(self, config: InstallerConfig) -> InstallerResult:
        """Create Linux AppImage. Refactored for JPL Rule #4."""
        output_path = (
            config.output_dir
            / f"{config.metadata.name}-{config.metadata.version}.AppImage"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir) / "AppDir"
            app_dir.mkdir()
            self._setup_appdir(app_dir, config)
            subprocess.run(["appimagetool", str(app_dir), str(output_path)], check=True)

        return InstallerResult(
            True, config.format, output_path, size_bytes=output_path.stat().st_size
        )

    def _setup_appdir(self, app_dir: Path, config: InstallerConfig) -> None:
        """Helper to populate AppDir structure."""
        shutil.copy2(config.binary_path, app_dir / config.binary_path.name)
        run_script = app_dir / "AppRun"
        run_script.write_text(
            f'#!/bin/bash\nexec $APPDIR/{config.binary_path.name} "$@"\n'
        )
        run_script.chmod(0o755)


def create_installer(binary_path: Path, platform_name: str, output_dir: Path) -> Path:
    """Helper for build_binary.py integration."""
    config = InstallerConfig(
        format=InstallerFormat.NSIS
        if platform_name == "windows"
        else InstallerFormat.DMG
        if platform_name == "darwin"
        else InstallerFormat.DEB,
        binary_path=binary_path,
        output_dir=output_dir,
    )
    result = InstallerPackager().package(config)
    if not result.success:
        raise RuntimeError(result.error)
    return result.output_path or Path("")


if __name__ == "__main__":
    pass
