@echo off
setlocal

set "PROJECT_DIR=%~dp0"
pushd "%PROJECT_DIR%"

echo [1/2] Stopping GUI window if running...
taskkill /FI "WINDOWTITLE eq FaceID GUI*" /T /F >nul 2>nul

echo [2/2] Stopping Docker services...
docker compose down
if errorlevel 1 (
  echo Failed to stop Docker services.
  popd
  exit /b 1
)

echo Done. GUI and backend stack are stopped.
popd
exit /b 0
