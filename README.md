# FaceID Browser Project (Windows)

This document explains how to run the FaceID system in full Docker mode, using a browser-based Gradio frontend with a FastAPI backend and PostgreSQL.

## 1) Overview

The stack includes 3 services:

- `postgres`: stores settings, attendance logs, and metadata.
- `app`: FastAPI backend for detection/recognition models and websocket inference.
- `frontend`: browser UI built with Gradio.

Default ports:

- Frontend: `http://localhost:7860`
- Backend API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`

## 2) Prerequisites

- Docker Desktop installed and running.
- Python 3.11 is recommended if you want to run scripts locally outside Docker.
- Model weights available in the `weights/` directory.

Reference models:

| Model              | Link                                                                                                      | Size    |
| ------------------ | --------------------------------------------------------------------------------------------------------- | ------- |
| SCRFD 10G          | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx)     | 16.1 MB |
| SCRFD 500M         | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx)   | 2.4 MB  |
| ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 13 MB   |
| ArcFace ResNet-50  | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB  |

## 3) Quick Start (One Command)

From the project folder:

```powershell
cd .\project
docker compose up -d --build
```

Then open the browser UI:

```powershell
start http://localhost:7860
```

Check status:

```powershell
docker compose ps
```

## 4) Step-by-Step Startup

If you want to start each service separately for easier debugging:

```powershell
cd .\project
docker compose build
docker compose up -d postgres
docker compose up -d app
docker compose up -d frontend
```

Readiness checks:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/api/infer/status
Invoke-WebRequest http://localhost:7860
```

## 5) Using the Browser UI

1. Open `http://localhost:7860`
2. Allow camera permission in the browser.
3. In `Video Source`, keep `0` for browser webcam mode.
4. Click `Start`.
5. Monitor `Status`, `Live Stream`, `Attendance List`, and `Unknown List`.

## 6) Stop or Reset the System

Stop all services:

```powershell
docker compose down
```

Stop and remove database volume (use with caution):

```powershell
docker compose down -v
```

## 7) Useful Debug Commands

Live logs:

```powershell
docker compose logs -f app frontend
```

Recent logs:

```powershell
docker compose logs --since 5m frontend
docker compose logs --since 5m app
```

Rebuild only frontend after editing `gui.py`:

```powershell
docker compose up -d --build frontend
```

## 8) Camera Notes (Windows + Docker)

- In full Docker mode, browser webcam is preferred over direct USB webcam access inside containers.
- Camera index `0/1` in Docker frontend does not always map correctly to the host webcam.
- Most stable workflow on Windows: open UI in browser, allow camera permission, keep source `0`, then click `Start`.

## 9) Data Persistence

- `assets/` is mounted into backend container for captures.
- `database/face_database/` is mounted for face metadata/index files.
- PostgreSQL uses the `postgres_data` volume.

## 10) Important Note

You cannot auto-open a browser tab directly from `docker compose up -d --build` itself because opening a browser is a host-side action. Open it manually with `start http://localhost:7860`, or use a host script (`.bat`/`.ps1`) to add that step.
