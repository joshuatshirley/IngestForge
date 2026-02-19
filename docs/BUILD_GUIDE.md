# IngestForge Build Guide

**US-3001.1: Portable PyInstaller Build**
**Epic AC-08: Cross-Platform Build Guide**
**Last Updated**: 2026-02-18T23:45:00Z

---

## 📋 Prerequisites

### Required

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Runtime and build scripts |
| **Node.js** | 18+ | Frontend build |
| **PyInstaller** | 6.3.0 | Binary packaging |

### Optional

| Tool | Version | Purpose |
|------|---------|---------|
| **UPX** | Latest | Binary compression (50-70% size reduction) |
| **Nuitka** | 2.0.6+ | Alternative build tool (performance) |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd frontend && npm install && cd ..
```

### 2. Build for Current Platform

```bash
# Basic build (Windows/macOS/Linux)
python scripts/build_binary.py

# Expected output: dist/ingestforge.exe (Windows) or dist/ingestforge (Unix)
```

### 3. Verify Build

```bash
# Test binary
./dist/ingestforge --version
./dist/ingestforge --help

# Run smoke tests
python scripts/build_binary.py --verify
```

---

## 🔧 Build Options

### Basic Usage

```bash
# Build for current platform
python scripts/build_binary.py

# Build with directory structure (not single-file)
python scripts/build_binary.py --onedir

# Debug build (verbose output)
python scripts/build_binary.py --debug
```

### Size Optimization 

```bash
# Build with maximum compression
python scripts/build_binary.py --upx-level max

# Build without UPX compression (faster build, larger binary)
python scripts/build_binary.py --no-upx

# Build with balanced compression (default)
python scripts/build_binary.py --upx-level balanced
```

### Installer Generation 

```bash
# Create installer after build
python scripts/build_binary.py --create-installer

# Expected output:
# - Windows: dist/IngestForge-Setup.exe
# - macOS: dist/IngestForge.dmg
# - Linux: dist/ingestforge.deb
```

### Post-Build Verification 

```bash
# Build with smoke tests (default)
python scripts/build_binary.py

# Skip smoke tests (faster)
python scripts/build_binary.py --no-verify
```

---

## 📦 Platform-Specific Instructions

### Windows

**Prerequisites**:
```powershell
# Install Chocolatey (package manager)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install UPX
choco install upx

# Install Python 3.10+
choco install python --version=3.10.0

# Install Node.js 18+
choco install nodejs --version=18.0.0
```

**Build**:
```powershell
# Build single-file executable
python scripts\build_binary.py

# Output: dist\ingestforge.exe (~150-200MB)
```

**Code Signing** (optional):
```powershell
# Sign executable (requires certificate)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\ingestforge.exe
```

**Installer Creation**:
```powershell
# Requires NSIS or Inno Setup
python scripts\build_binary.py --create-installer
```

---

### macOS

**Prerequisites**:
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install UPX
brew install upx

# Install Python 3.10+
brew install python@3.10

# Install Node.js 18+
brew install node@18
```

**Build**:
```bash
# Build Mach-O executable
python scripts/build_binary.py

# Output: dist/ingestforge (~150-200MB)
```

**Code Signing & Notarization**:
```bash
# Sign with Apple Developer ID
codesign --force --sign "Developer ID Application: Your Name" dist/ingestforge

# Notarize (required for macOS 10.15+)
xcrun notarytool submit dist/ingestforge.zip \
  --apple-id "your@email.com" \
  --team-id "TEAM_ID" \
  --password "app-specific-password" \
  --wait

# Staple notarization ticket
xcrun stapler staple dist/ingestforge
```

**DMG Creation**:
```bash
# Install create-dmg
brew install create-dmg

# Create installer
python scripts/build_binary.py --create-installer
# Output: dist/IngestForge.dmg
```

---

### Linux

**Prerequisites** (Ubuntu/Debian):
```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip nodejs npm upx

# Install Python dependencies
pip3 install -r requirements.txt
```

**Build**:
```bash
# Build ELF executable
python3 scripts/build_binary.py

# Output: dist/ingestforge (~150-200MB)
```

**Permissions**:
```bash
# Make executable
chmod +x dist/ingestforge
```

**Debian Package Creation**:
```bash
# Install fpm (package builder)
sudo apt-get install -y ruby ruby-dev build-essential
sudo gem install fpm

# Create .deb package
python scripts/build_binary.py --create-installer
# Output: dist/ingestforge.deb
```

**RPM Package Creation** (RHEL/CentOS/Fedora):
```bash
# Install rpm-build
sudo dnf install -y rpm-build

# Create .rpm package
fpm -s dir -t rpm -n ingestforge -v 1.0.0 dist/ingestforge=/usr/bin/
```

---

## 🧪 Testing Builds

### Manual Testing

```bash
# Version check
./dist/ingestforge --version
# Expected: "IngestForge v1.0.0"

# Help command
./dist/ingestforge --help
# Expected: CLI help output

# Basic functionality
./dist/ingestforge ingest sample.txt --dry-run
# Expected: Dry-run success message

# API server
./dist/ingestforge serve --port 8001 &
curl http://localhost:8001/health
# Expected: {"status": "healthy"}
```

### Automated Smoke Tests

```bash
# Run all smoke tests
pytest tests/integration/test_binary_smoke.py --binary=dist/ingestforge

# Expected: 5 tests passed
```

---

## 📊 Binary Size Targets 

| Platform | Target | Typical | With UPX (max) |
|----------|--------|---------|----------------|
| **Windows** | <200MB | ~180MB | ~120MB |
| **macOS** | <200MB | ~175MB | ~115MB |
| **Linux** | <200MB | ~170MB | ~110MB |

**Size Breakdown**:
- Python runtime: ~50MB
- Dependencies (numpy, pandas, etc.): ~80MB
- ML libraries (sentence-transformers): ~30MB
- Frontend (React bundle): ~5MB
- Application code: ~5MB

**Optimization Tips**:
1. Use `--upx-level max` for maximum compression
2. Exclude dev dependencies (already done automatically)
3. Use `--onedir` mode (faster but larger)
4. Lazy-load heavy ML models

---

## 🔄 CI/CD Integration

### GitHub Actions

Create `.github/workflows/build-binaries.yml`:

```yaml
name: Build Binaries

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install UPX
        run: choco install upx

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build binary
        run: python scripts/build_binary.py --upx-level max --create-installer

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: ingestforge-windows
          path: |
            dist/ingestforge.exe
            dist/IngestForge-Setup.exe

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install UPX
        run: brew install upx

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build binary
        run: python scripts/build_binary.py --upx-level max --create-installer

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: ingestforge-macos
          path: |
            dist/ingestforge
            dist/IngestForge.dmg

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install UPX
        run: sudo apt-get install -y upx

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build binary
        run: python scripts/build_binary.py --upx-level max --create-installer

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: ingestforge-linux
          path: |
            dist/ingestforge
            dist/ingestforge.deb
```

---

## 🐛 Troubleshooting

### "PyInstaller not found"

**Problem**: `ModuleNotFoundError: No module named 'PyInstaller'`

**Solution**:
```bash
pip install PyInstaller==6.3.0
```

### "UPX not found, binaries won't be compressed"

**Problem**: UPX not installed, binary >200MB

**Solution**:
- Windows: `choco install upx`
- macOS: `brew install upx`
- Linux: `sudo apt-get install upx`

### "Frontend build failed: 'out' directory not found"

**Problem**: Node.js dependencies not installed

**Solution**:
```bash
cd frontend
npm install
npm run build
cd ..
```

### Binary >200MB after build

**Problem**: Binary exceeds size target

**Solutions**:
1. Use `--upx-level max` for maximum compression
2. Check `build_metrics.json` for size trends
3. Verify dev dependencies are excluded (automatic)

### "Cross-compilation not supported"

**Problem**: Trying to build for different platform

**Solution**: Build on the target platform (use CI/CD or VMs)

### macOS "App is damaged and can't be opened"

**Problem**: Binary not signed/notarized

**Solution**:
```bash
# Sign the binary
codesign --force --sign "Developer ID Application: Your Name" dist/ingestforge

# Or allow unsigned apps (development only)
xattr -cr dist/ingestforge
```

---

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| **Frontend build** | 2-3 min | Includes npm install |
| **PyInstaller build** | 3-5 min | Single-file mode |
| **UPX compression (balanced)** | 1-2 min | 50-60% reduction |
| **UPX compression (max)** | 2-4 min | 60-70% reduction |
| **Total build time** | **8-12 min** | With all optimizations |

**Binary startup time**:
- Without UPX: ~500ms
- With UPX (balanced): ~800ms
- With UPX (max): ~1200ms

---

## 🔐 Security Considerations

### Code Signing

**Windows**:
- Requires code signing certificate ($200-400/year)
- Use `signtool` from Windows SDK
- Prevents SmartScreen warnings

**macOS**:
- Requires Apple Developer ID ($99/year)
- Use `codesign` and notarization
- Required for macOS 10.15+ (Catalina)

**Linux**:
- Optional (no system-level requirement)
- Can use GPG signatures for packages

### Best Practices

1. ✅ Sign all production binaries
2. ✅ Verify checksums (SHA256)
3. ✅ Use HTTPS for downloads
4. ✅ Publish checksums separately
5. ✅ Keep signing certificates secure

---

## 📚 Additional Resources

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [UPX Homepage](https://upx.github.io/)
- [Nuitka Documentation](https://nuitka.net/)
- [macOS Code Signing Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Windows Code Signing Guide](https://docs.microsoft.com/en-us/windows/win32/seccrypto/signtool)

---

## 🆘 Getting Help

**Issues**: https://github.com/anthropics/ingestforge/issues
**Discussions**: https://github.com/anthropics/ingestforge/discussions
**Email**: support@ingestforge.example.com

---

**Build Guide Version**: 1.0.0
**US-3001.1**: Portable PyInstaller Build
**Last Updated**: 2026-02-18T23:45:00Z
