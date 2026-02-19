# Troubleshooting

This guide provides solutions to common issues encountered during installation and use of IngestForge.

## 🛠️ Installation Issues

### 1. Python version mismatch
*   **Error**: `Python 3.10+ required but not found`
*   **Solution**: Ensure you have Python 3.10 or higher installed. Run `python --version` to check. If you have multiple versions, use `python3` instead.

### 2. missing `python3-venv` (Linux)
*   **Error**: `The virtual environment was not created successfully`
*   **Solution**: On Ubuntu/Debian systems, install the venv package: `sudo apt update && sudo apt install python3-venv`.

### 3. Node.js not found
*   **Warning**: `Node.js not found (frontend features disabled)`
*   **Solution**: Download and install Node.js 18+ from [nodejs.org](https://nodejs.org/). This is required if you want to use the Web Portal.

## ⚙️ Runtime Issues

### 4. Search fails with "Engine offline"
*   **Cause**: The background API server is not running.
*   **Solution**: Ensure you have started the backend. Run `./Start_API.bat` or `ingestforge server start`.

### 5. Low Memory Warnings
*   **Issue**: System feels slow or crashes during ingestion.
*   **Cause**: Ingesting large batches of PDFs requires significant RAM.
*   **Solution**: Increase your swap space or ingest documents in smaller batches. IngestForge recommends at least 4GB RAM for optimal performance.

### 6. PDF Processing Errors
*   **Issue**: "Failed to extract text from file.pdf"
*   **Cause**: The PDF may be password-protected or use an unsupported encryption.
*   **Solution**: Remove password protection before ingesting. If the file is a scanned image, ensure OCR is enabled in your configuration.

## 🌐 Web Portal Issues

### 7. Cannot connect to localhost:3000
*   **Solution**: Verify the frontend is running. Look for the `Start_Frontend.bat` terminal window. Ensure no other application is using port 3000.

---

## 🆘 Still Need Help?

If your issue isn't listed here:
1.  Check the [**GitHub Issues**](https://github.com/ingestforge/ingestforge/issues) page.
2.  Run `ingestforge monitor health` to see a diagnostic report.
3.  Join our Discord community for real-time support.
