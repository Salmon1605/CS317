# Face Re-ID Project Run Guide (Windows)

Project source is in [project](project).

## Prerequisites

- Python 3.11 installed (`py -3.11` available)
- Docker Desktop installed and running

---

## Model checkpoint

- Download model checkpoint and create directory weights/
  | Model | Link | Dung lượng |
  |-------|------|-----------|
  | SCRFD 10G (detection) | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx) | 16.1 MB |
  | SCRFD 500M (nhẹ) | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx) | 2.4 MB |
  | ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 13 MB |
  | ArcFace ResNet-50 | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB |

---

## Method 1: Use [project/start_all.bat](project/start_all.bat)

### 1) Prepare a venv (one-time setup)

Run these commands in [project](project):

```powershell
cd .\project
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install --upgrade pip
.\venv311\Scripts\python.exe -m pip install -r .\requirements-gui.txt
```

Note: [project/start_all.bat](project/start_all.bat) prefers `venv311`; if it does not exist, it falls back to `venv`.

### 2) Start the full system

```powershell
cd .\project
.\start_all.bat
```

The script will:

- Build and run backend/database with `docker compose up -d`
- Wait for the backend to become ready
- Launch the GUI (`gui.py`)

### 3) Stop the system

```powershell
cd .\project
.\stop_all.bat
```

---

## Method 2: Do not use [project/start_all.bat](project/start_all.bat)

### 1) Create and install a venv for GUI

```powershell
cd .\project
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install --upgrade pip
.\venv311\Scripts\python.exe -m pip install -r .\requirements-gui.txt
```

### 2) Start backend with Docker Compose

```powershell
cd .\project
docker compose up -d --build
```

Quick check:

```powershell
docker compose ps
```

### 3) Run GUI locally

```powershell
cd .\project
.\venv311\Scripts\python.exe .\gui.py
```

### 4) Stop backend

```powershell
cd .\project
docker compose down
```

---

## Notes

- Backend runs in Docker on port `8000`.
- GUI runs locally and connects to backend via `http://localhost:8000` and `ws://localhost:8000/ws/infer`.
- If you want to install the full dependency set for local development (backend + GUI):

```powershell
cd .\project
.\venv311\Scripts\python.exe -m pip install -r .\requirements.txt
```
