@echo off
setlocal

set "PROJECT_DIR=%~dp0"
pushd "%PROJECT_DIR%"

echo [1/3] Starting Docker services...
docker compose up -d --build
if errorlevel 1 (
  echo Failed to start Docker services.
  popd
  exit /b 1
)

echo [2/3] Waiting for backend to respond...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $ok=$false; for($i=0; $i -lt 60; $i++){ try { $r=Invoke-RestMethod -Uri 'http://localhost:8000/api/infer/status' -Method Get -TimeoutSec 2; if($null -ne $r){ $ok=$true; break } } catch {} Start-Sleep -Seconds 1 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo Backend did not become ready in time.
  popd
  exit /b 1
)

set "PYTHON_EXE=%PROJECT_DIR%venv311\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=%PROJECT_DIR%venv\Scripts\python.exe"
)

if not exist "%PYTHON_EXE%" (
  echo Python venv not found. Expected:
  echo   %PROJECT_DIR%venv311\Scripts\python.exe
  echo or
  echo   %PROJECT_DIR%venv\Scripts\python.exe
  popd
  exit /b 1
)

echo [3/3] Launching GUI...
start "FaceID GUI" cmd /c ""%PYTHON_EXE%" "%PROJECT_DIR%gui.py""

echo Done. Backend is running and GUI has been started.
popd
exit /b 0
