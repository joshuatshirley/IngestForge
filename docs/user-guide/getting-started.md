# Getting Started

This guide will help you install IngestForge and run your first research query in under 10 minutes.

## 📋 Prerequisites

Before installing, ensure your system meets the minimum software requirements:

*   **Python**: 3.10 or higher
*   **Node.js**: 18.0 or higher (required for the Web Portal)
*   **Hardware**: 2GB+ RAM, 5GB+ Disk Space

## 🛠️ Installation

IngestForge provides a **One-Command Installer** that automates virtual environment creation and dependency management.

### Windows (PowerShell)
```powershell
./install.bat
```

### macOS / Linux
```bash
curl -sSL https://raw.githubusercontent.com/ingestforge/ingestforge/main/scripts/install.sh | bash
```

## 🪄 Initial Configuration

After the installer completes, the **Enhanced Setup Wizard** will launch automatically. It will:

1.  **Detect Hardware**: Check your CPU/RAM to recommend the best model preset.
2.  **Download Models**: Fetch the required embedding models (typically ~500MB).
3.  **Generate Config**: Create your local `.ingestforge/config.json`.

If you need to run the wizard manually later:
```bash
ingestforge setup
```

## 🚀 Running the Demo

To verify your installation and see IngestForge in action, run the demo command:

```bash
ingestforge demo
```

This will download a sample dataset, ingest the documents, and show you how retrieval and synthesis work.

## 🌐 Opening the Web Portal

If you prefer a graphical interface, you can start the Forge Portal:

```bash
# Start the API and Frontend
./IngestForgeGUI.bat
```
Then navigate to `http://localhost:3000` in your browser.
