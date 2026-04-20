import logging
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
# class DummyDB:
#     async def init_pool(self): pass
#     async def close_pool(self): pass
#     async def load_settings(self): return {}
#     async def save_setting(self, *args, **kwargs): pass
#     async def log_attendance(self, *args, **kwargs): return 1
#     async def log_unknown(self, *args, **kwargs): return 1
#     async def get_attendance(self, *args, **kwargs): return []
#     async def get_unknowns(self, *args, **kwargs): return []
#     async def get_stats(self): return {}

# db = DummyDB()

from database import FaceDatabase
from models import ArcFace, FaceDetector, create_detector
from utils.logging import setup_logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"onnxruntime.*")

setup_logging(log_to_file=True)
logger = logging.getLogger("api")


class AppState:
    detector: Optional[FaceDetector] = None
    recognizer: Optional[ArcFace] = None
    face_db: Optional[FaceDatabase] = None

    # Settings (loaded from DB on startup)
    det_weight: str = "./weights/det_10g.onnx"
    rec_weight: str = "./weights/w600k_mbf.onnx"
    confidence_thresh: float = 0.5
    similarity_thresh: float = 0.4
    db_path: str = "./database/face_database"
    faces_dir: str = "./assets/faces"
    unknown_debounce_sec: int = 5
    known_debounce_min: int = 1
    infer_enabled: bool = False


state = AppState()




def initialize_models():
    try:
        logger.info("Loading models...")
        state.detector = create_detector(
            state.det_weight,
            input_size=(640, 640),
            conf_thres=state.confidence_thresh,
        )
        state.recognizer = ArcFace(state.rec_weight)
        state.face_db = FaceDatabase(embedding_size=state.recognizer.embedding_size, db_path=state.db_path)
        if not state.face_db.load():
            logger.info("Face database not found, building...")
            build_database()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


def build_database():
    if not os.path.exists(state.faces_dir):
        logger.warning(f"Faces directory {state.faces_dir} does not exist. Creating empty DB.")
        state.face_db.save()
        return
    embeddings, names = [], []
    for filename in sorted(os.listdir(state.faces_dir)):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue
        name = filename.rsplit(".", 1)[0]
        image_path = os.path.join(state.faces_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        try:
            bboxes, kpss = state.detector.detect(image, max_num=1)
            if len(kpss) == 0:
                continue
            embedding = state.recognizer.get_embedding(image, kpss[0])
            embeddings.append(embedding)
            names.append(name)
            logger.info(f"Added face for: {name}")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    if embeddings:
        state.face_db.add_faces_batch(embeddings, names)
    state.face_db.save()
    logger.info("Database built and saved.")


def run_inference_on_frame(frame: np.ndarray) -> list[dict]:
    bboxes, kpss = state.detector.detect(frame, max_num=0)
    results = []
    if len(bboxes) == 0:
        return results

    embeddings, processed_bboxes = [], []
    for bbox, kps in zip(bboxes, kpss):
        *bbox_coords, _ = bbox.astype(np.int32)
        embedding = state.recognizer.get_embedding(frame, kps)
        embeddings.append(embedding)
        processed_bboxes.append(bbox_coords)

    if not embeddings:
        return results

    search_results = state.face_db.batch_search(embeddings, state.similarity_thresh)
    for bbox, (name, similarity) in zip(processed_bboxes, search_results):
        results.append(
            {
                "bbox": [int(b) for b in bbox],
                "name": name,
                "similarity": float(similarity),
            }
        )
    return results


async def _apply_db_settings(raw: dict):
    """Apply a dict of string->string settings from DB into AppState."""
    if "det_weight" in raw:
        state.det_weight = raw["det_weight"]
    if "rec_weight" in raw:
        state.rec_weight = raw["rec_weight"]
    if "confidence_thresh" in raw:
        state.confidence_thresh = float(raw["confidence_thresh"])
    if "similarity_thresh" in raw:
        state.similarity_thresh = float(raw["similarity_thresh"])
    if "unknown_debounce_sec" in raw:
        state.unknown_debounce_sec = int(raw["unknown_debounce_sec"])
    if "known_debounce_min" in raw:
        state.known_debounce_min = int(raw["known_debounce_min"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    await db.init_pool()
    raw_settings = await db.load_settings()
    if raw_settings:
        await _apply_db_settings(raw_settings)
        logger.info("Settings loaded from PostgreSQL.")
    initialize_models()
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    await db.close_pool()


app = FastAPI(title="Face Re-ID API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("./assets/faces", exist_ok=True)
os.makedirs("./assets/captures/attendance", exist_ok=True)
os.makedirs("./assets/captures/unknown", exist_ok=True)
app.mount("/faces",    StaticFiles(directory="./assets/faces"),    name="faces")
app.mount("/captures", StaticFiles(directory="./assets/captures"), name="captures")


def _save_capture(subfolder: str, name_prefix: str, image_bytes: bytes) -> str:
    """Save jpeg bytes to ./assets/captures/{subfolder}/, return URL path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = name_prefix.replace("/", "_").replace("\\", "_")
    filename = f"{ts}_{safe}.jpg"
    filepath = os.path.join("./assets/captures", subfolder, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return f"/captures/{subfolder}/{filename}"


# ═══════════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsUpdate(BaseModel):
    det_weight: Optional[str] = None
    rec_weight: Optional[str] = None
    confidence_thresh: Optional[float] = None
    similarity_thresh: Optional[float] = None
    unknown_debounce_sec: Optional[int] = None
    known_debounce_min: Optional[int] = None


@app.get("/api/settings")
async def get_settings():
    return {
        "det_weight": state.det_weight,
        "rec_weight": state.rec_weight,
        "confidence_thresh": state.confidence_thresh,
        "similarity_thresh": state.similarity_thresh,
        "unknown_debounce_sec": state.unknown_debounce_sec,
        "known_debounce_min": state.known_debounce_min,
    }


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    reload_needed = False

    if settings.det_weight is not None and settings.det_weight != state.det_weight:
        state.det_weight = settings.det_weight
        await db.save_setting("det_weight", state.det_weight)
        reload_needed = True

    if settings.rec_weight is not None and settings.rec_weight != state.rec_weight:
        state.rec_weight = settings.rec_weight
        await db.save_setting("rec_weight", state.rec_weight)
        reload_needed = True

    if settings.confidence_thresh is not None:
        state.confidence_thresh = settings.confidence_thresh
        await db.save_setting("confidence_thresh", str(state.confidence_thresh))
        if state.detector:
            state.detector.conf_thres = state.confidence_thresh

    if settings.similarity_thresh is not None:
        state.similarity_thresh = settings.similarity_thresh
        await db.save_setting("similarity_thresh", str(state.similarity_thresh))

    if settings.unknown_debounce_sec is not None:
        state.unknown_debounce_sec = settings.unknown_debounce_sec
        await db.save_setting("unknown_debounce_sec", str(state.unknown_debounce_sec))

    if settings.known_debounce_min is not None:
        state.known_debounce_min = settings.known_debounce_min
        await db.save_setting("known_debounce_min", str(state.known_debounce_min))

    if reload_needed:
        initialize_models()

    return {"status": "success", "message": "Settings updated"}


# ═══════════════════════════════════════════════════════════════════════════════
# Inference control
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/infer/start")
async def start_infer():
    state.infer_enabled = True
    return {"status": "success", "enabled": True}


@app.post("/api/infer/stop")
async def stop_infer():
    state.infer_enabled = False
    return {"status": "success", "enabled": False}


@app.get("/api/infer/status")
async def infer_status():
    return {"enabled": state.infer_enabled}


@app.post("/api/infer/frame")
async def infer_frame(image: UploadFile = File(...)):
    if not state.detector or not state.recognizer or not state.face_db:
        raise HTTPException(status_code=500, detail="Models not loaded")

    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    if not state.infer_enabled:
        return {"results": [], "enabled": False}

    try:
        return {"results": run_inference_on_frame(frame), "enabled": True}
    except Exception as e:
        logger.error(f"Inference error (HTTP frame): {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Face database
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/database/update")
async def update_database_api():
    try:
        state.face_db = FaceDatabase(embedding_size=state.recognizer.embedding_size, db_path=state.db_path)
        build_database()
        return {"status": "success", "message": "Database updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    weights_dir = "./weights"
    if not os.path.exists(weights_dir):
        return {"models": []}
    try:
        models = sorted(f for f in os.listdir(weights_dir) if f.endswith(".onnx"))
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"models": []}


# ═══════════════════════════════════════════════════════════════════════════════
# Attendance & Unknown logs
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/attendance/log", status_code=201)
async def post_attendance_log(
    name: str = Form(...),
    similarity: float = Form(0.0),
    image: Optional[UploadFile] = File(None),
):
    image_path = None
    if image and image.filename:
        img_bytes = await image.read()
        image_path = _save_capture("attendance", name, img_bytes)
    row_id = await db.log_attendance(name, similarity, image_path)
    return {"status": "logged", "id": row_id, "image_path": image_path}


@app.post("/api/unknown/log", status_code=201)
async def post_unknown_log(
    image: Optional[UploadFile] = File(None),
):
    image_path = None
    if image and image.filename:
        img_bytes = await image.read()
        image_path = _save_capture("unknown", "unknown", img_bytes)
    row_id = await db.log_unknown(image_path)
    return {"status": "logged", "id": row_id, "image_path": image_path}


@app.get("/api/attendance")
async def list_attendance(
    name: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(200, le=1000),
):
    rows = await db.get_attendance(limit=limit, name=name, date_from=date_from, date_to=date_to)
    # Convert datetime to ISO string for JSON
    for r in rows:
        if isinstance(r.get("detected_at"), datetime):
            r["detected_at"] = r["detected_at"].isoformat()
    return {"data": rows, "count": len(rows)}


@app.get("/api/unknowns")
async def list_unknowns(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(200, le=1000),
):
    rows = await db.get_unknowns(limit=limit, date_from=date_from, date_to=date_to)
    for r in rows:
        if isinstance(r.get("detected_at"), datetime):
            r["detected_at"] = r["detected_at"].isoformat()
    return {"data": rows, "count": len(rows)}


@app.get("/api/stats")
async def get_stats():
    return await db.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket inference
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/infer")
async def websocket_infer(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected for inference.")
    try:
        while True:
            data = await websocket.receive_bytes()

            if not state.detector or not state.recognizer or not state.face_db:
                await websocket.send_json({"error": "Models not loaded"})
                continue

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            if not state.infer_enabled:
                await websocket.send_json({"results": []})
                continue

            try:
                await websocket.send_json({"results": run_inference_on_frame(frame)})
            except Exception as e:
                logger.error(f"Inference error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)