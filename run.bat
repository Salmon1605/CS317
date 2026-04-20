@echo off
echo Starting FastAPI Backend...
start cmd /k "python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for FastAPI to start...
timeout /t 5 /nobreak >nul

echo Starting Browser Frontend (Gradio)...
start cmd /k "python gui.py"

echo Both services started!