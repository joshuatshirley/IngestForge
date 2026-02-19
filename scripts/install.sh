#!/bin/bash
# IngestForge One-Command Installer (Unix/macOS/Linux)
#
# Usage:
#   bash install.sh [--no-wizard] [--verbose]
#
# JPL Power of Ten Compliance:
# - Rule #2: Bounded error checks via 'set -e'
# - Rule #7: All commands checked for success

set -e  # Exit on error
set -u  # Exit on undefined variable

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "                    IngestForge Installer (Unix)"
echo "========================================================================"

# Check Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Error: Python 3.10+ required but not found.${NC}"
    echo ""
    echo "Install Python:"
    echo "  - macOS: brew install python@3.11"
    echo "  - Ubuntu/Debian: sudo apt install python3.11"
    echo "  - Fedora/RHEL: sudo dnf install python3.11"
    echo "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Found Python ${PYTHON_VERSION}${NC}"

# Check for python3-venv (Common blocker on clean Ubuntu)
if ! python3 -m venv --help &> /dev/null; then
    echo -e "${RED}✗ Error: 'python3-venv' package is missing.${NC}"
    echo "  On Ubuntu/Debian, run: sudo apt update && sudo apt install python3-venv"
    exit 1
fi

# Check Node.js (optional but recommended)
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠ Warning: Node.js not found (frontend features disabled)${NC}"
    echo "  Install Node.js 18+ from: https://nodejs.org/"
else
    NODE_VERSION=$(node --version 2>&1)
    echo -e "${GREEN}✓ Found Node.js ${NODE_VERSION}${NC}"
fi

# Run Python installer
echo ""
echo "Running Python installer..."
echo ""

python3 "$(dirname "$0")/install.py" "$@"

INSTALL_EXIT_CODE=$?

if [ $INSTALL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}✓ Installation successful!${NC}"
    echo "========================================================================"
    echo ""
    echo "Optional: Add IngestForge to your PATH"
    echo ""
    read -p "Add 'ingestforge' alias to your shell? [Y/n] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        # Determine install directory and venv path
        INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
        VENV_PYTHON="$INSTALL_DIR/venv/bin/python3"

        # Determine shell config file
        if [ -n "${BASH_VERSION:-}" ]; then
            SHELL_CONFIG="$HOME/.bashrc"
        elif [ -n "${ZSH_VERSION:-}" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        else
            SHELL_CONFIG="$HOME/.profile"
        fi

        # Add alias if not already present
        if ! grep -q "alias ingestforge=" "$SHELL_CONFIG" 2>/dev/null; then
            echo "" >> "$SHELL_CONFIG"
            echo "# IngestForge alias (added by installer)" >> "$SHELL_CONFIG"
            echo "alias ingestforge='$VENV_PYTHON -m ingestforge.cli.main'" >> "$SHELL_CONFIG"

            echo -e "${GREEN}✓ Added alias to $SHELL_CONFIG${NC}"
            echo ""
            echo "Restart your shell or run: source $SHELL_CONFIG"
        else
            echo -e "${YELLOW}⚠ Alias already exists in $SHELL_CONFIG${NC}"
        fi
    fi

    echo ""
    echo "Try: ingestforge --help"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo -e "${RED}✗ Installation failed (exit code: $INSTALL_EXIT_CODE)${NC}"
    echo "========================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure Python 3.10+ is installed"
    echo "  2. Check you have write permissions"
    echo "  3. Try with --verbose flag for more details"
    echo "  4. See docs: https://docs.ingestforge.io/troubleshooting"
    echo ""
    exit $INSTALL_EXIT_CODE
fi
