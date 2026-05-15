@echo off
echo.
echo  ============================================
echo   Gemma Chat -- Local Server
echo  ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from https://python.org
    pause & exit /b 1
)

:: Install deps
echo  Installing dependencies...
pip install -r requirements.txt --quiet

:: Check .env
if not exist .env (
    echo.
    echo  ERROR: .env file not found.
    echo  Steps:
    echo    1. Run:  modal deploy modal_app.py
    echo    2. Copy the endpoint URLs from the output
    echo    3. Copy .env.example to .env and fill in the URLs
    echo.
    pause & exit /b 1
)

echo  Starting server at http://localhost:8000
echo  Press Ctrl+C to stop.
echo.
python server.py
pause
