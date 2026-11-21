@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Python environment not found. Follow the setup steps in README.md.
    pause
    exit /b 1
)
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8502 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    start "Sports Predictor engine" /min cmd /c "venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8502"
    timeout /t 7 /nobreak >nul
)
start "" msedge --app=http://localhost:8502
