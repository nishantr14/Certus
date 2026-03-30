@echo off
title CertusDoc Launcher

echo ========================================
echo   CertusDoc - Starting Services...
echo ========================================
echo.

:: Check if uvicorn is available
where uvicorn >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uvicorn not found. Please install requirements:
    echo         pip install -r requirements.txt
    pause
    exit /b 1
)

:: Add Tesseract to PATH if not already there
set "TESS_PATH=C:\Program Files\Tesseract-OCR"
if exist "%TESS_PATH%\tesseract.exe" (
    set "PATH=%PATH%;%TESS_PATH%"
    echo [OK] Tesseract OCR found
) else (
    echo [WARNING] Tesseract OCR not found - OCR features will be disabled
)

:: Add Poppler to PATH if not already there
set "POPPLER_PATH=C:\poppler\poppler-24.08.0\Library\bin"
if exist "%POPPLER_PATH%\pdftoppm.exe" (
    set "PATH=%PATH%;%POPPLER_PATH%"
    echo [OK] Poppler found
) else (
    echo [WARNING] Poppler not found - PDF analysis will fail
)

:: Start the backend in background
echo [1/2] Starting backend API server on port 8000...
start "CertusDoc Backend" cmd /k "cd /d %~dp0 && set PATH=%PATH%;%TESS_PATH%;%POPPLER_PATH% && uvicorn api:app --host 0.0.0.0 --port 8000"

:: Wait a moment for backend to initialize
echo       Waiting for backend to initialize...
timeout /t 4 /nobreak >nul

:: Check if backend started (optional curl check)
echo       Backend should be running at http://localhost:8000
echo.

:: Start the frontend
echo [2/2] Starting frontend dev server...
cd /d %~dp0\landing
start "CertusDoc Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo   Both services are starting!
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo ========================================
echo.
echo   Close the two opened windows to stop the services.
echo   Press any key to close this launcher window...
pause >nul
