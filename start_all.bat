@echo off
setlocal

set "PROJECT_DIR=%~dp0"
pushd "%PROJECT_DIR%"

echo [1/3] Starting full Docker stack (postgres + backend + frontend)...
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

echo [3/3] Waiting for browser frontend...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $ok=$false; for($i=0; $i -lt 60; $i++){ try { $r=Invoke-WebRequest -Uri 'http://localhost:7860' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -ge 200){ $ok=$true; break } } catch {} Start-Sleep -Seconds 1 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo Frontend did not become ready in time.
  popd
  exit /b 1
)

start "" "http://localhost:7860"

echo Done. Full stack is running in Docker.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:7860
popd
exit /b 0
