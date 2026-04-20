@echo off
setlocal

set "PROJECT_DIR=%~dp0"
pushd "%PROJECT_DIR%"

echo [1/1] Stopping Docker services...
docker compose down
if errorlevel 1 (
  echo Failed to stop Docker services.
  popd
  exit /b 1
)

echo Done. Full Docker stack is stopped.
popd
exit /b 0
