import datetime
import io
import json
import os
import time
import atexit
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import requests
from websockets.sync.client import connect as ws_connect

API_BASE_URL = os.getenv("INTERNAL_API_BASE_URL", "http://localhost:8000/api")
WS_URL = os.getenv("INTERNAL_WS_URL", "ws://localhost:8000/ws/infer")
FACE_BASE_URL = os.getenv("PUBLIC_FACE_BASE_URL", "http://localhost:8000/faces")
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
LAST_INFER_ERROR = ""
WS_CLIENT = None


def close_ws_client() -> None:
    global WS_CLIENT
    if WS_CLIENT is None:
        return
    try:
        WS_CLIENT.close()
    except Exception:
        pass
    WS_CLIENT = None


def get_ws_client():
    global WS_CLIENT
    if WS_CLIENT is not None:
        return WS_CLIENT
    WS_CLIENT = ws_connect(WS_URL, max_size=8 * 1024 * 1024, open_timeout=2)
    return WS_CLIENT


def safe_api_get(path: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def safe_api_post(path: str, json_payload: dict[str, Any] | None = None, timeout: float = 3.0) -> bool:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=json_payload, timeout=timeout)
        return response.status_code < 400
    except Exception:
        return False


def infer_faces(frame_bgr: np.ndarray, infer_max_width: int = 1280) -> list[dict[str, Any]]:
    global LAST_INFER_ERROR
    LAST_INFER_ERROR = ""

    orig_h, orig_w = frame_bgr.shape[:2]

    if orig_w > infer_max_width:
        scale = infer_max_width / orig_w
        infer_w = infer_max_width
        infer_h = round(orig_h * scale)
        small_frame = cv2.resize(frame_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        small_frame = frame_bgr

    ret, buf = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        return []

    data: Any = None
    for _ in range(2):
        try:
            ws = get_ws_client()
            ws.send(buf.tobytes())
            data = json.loads(ws.recv())
            break
        except Exception:
            close_ws_client()

    if data is None:
        LAST_INFER_ERROR = "Khong ket noi duoc websocket infer"
        return []

    if isinstance(data, dict) and data.get("error"):
        LAST_INFER_ERROR = str(data.get("error"))
        return []

    results = data.get("results", []) if isinstance(data, dict) else []
    if scale != 1.0:
        inv = 1.0 / scale
        for face in results:
            face["bbox"] = [int(v * inv) for v in face.get("bbox", [])]
    return results


def crop_face(frame_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)
    return frame_rgb[y1:y2, x1:x2]


def encode_face(face_img_rgb: np.ndarray) -> bytes | None:
    try:
        face_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
        ret, buf = cv2.imencode(".jpg", face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ret else None
    except Exception:
        return None


def coerce_frame_to_rgb_array(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None

    if isinstance(frame, np.ndarray):
        arr = frame
    elif isinstance(frame, dict):
        frame_path = frame.get("path")
        if not frame_path:
            return None
        img_bgr = cv2.imread(str(Path(str(frame_path))))
        if img_bgr is None:
            return None
        arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        try:
            arr = np.asarray(frame)
        except Exception:
            return None

    if arr.size == 0:
        return None

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        return None

    return np.ascontiguousarray(arr)


def log_attendance(name: str, similarity: float, face_bytes: bytes | None = None) -> None:
    try:
        files = {"image": ("face.jpg", io.BytesIO(face_bytes), "image/jpeg")} if face_bytes else {}
        data = {"name": name, "similarity": str(float(similarity))}
        requests.post(f"{API_BASE_URL}/attendance/log", data=data, files=files, timeout=5)
    except Exception:
        pass


def log_unknown(face_bytes: bytes | None = None) -> None:
    try:
        files = {"image": ("face.jpg", io.BytesIO(face_bytes), "image/jpeg")} if face_bytes else {}
        requests.post(f"{API_BASE_URL}/unknown/log", files=files, timeout=5)
    except Exception:
        pass


def render_unknown_html(items: list[dict[str, Any]]) -> str:
    cards = []
    for item in items:
        cards.append(
            (
                '<div class="face-card unknown">'
                f'<img src="data:image/jpeg;base64,{item["image"]}" alt="unknown" />'
                f'<div class="meta">Unknown<br/><span>{item["time"]}</span></div>'
                "</div>"
            )
        )
    return '<div class="face-strip">' + "".join(cards) + "</div>"


def render_attendance_html(items: list[dict[str, Any]]) -> str:
    cards = []
    for item in items:
        db_image = f"{FACE_BASE_URL}/{item['name']}.jpg"
        cards.append(
            (
                '<div class="att-card">'
                f'<img class="db" src="{db_image}" alt="{item["name"]}" onerror="this.style.opacity=0.3" />'
                f'<img class="live" src="data:image/jpeg;base64,{item["image"]}" alt="live" />'
                '<div class="meta">'
                f'<div class="name">{item["name"]}</div>'
                f'<div class="sim">Sim: {item["similarity"]:.2f}</div>'
                f'<div class="time">{item["time"]}</div>'
                "</div>"
                "</div>"
            )
        )
    return '<div class="att-list">' + "".join(cards) + "</div>"


@dataclass
class PipelineState:
    camera_source: str = "0"
    running: bool = False
    cap: cv2.VideoCapture | None = None
    known_faces_history: dict[str, float] = field(default_factory=dict)
    last_unknown_seen: float = 0.0
    unknown_debounce_sec: int = 5
    known_debounce_min: int = 1
    unknown_items: list[dict[str, Any]] = field(default_factory=list)
    attendance_items: list[dict[str, Any]] = field(default_factory=list)
    last_tick: float = field(default_factory=time.time)
    use_browser_webcam: bool = False
    backend_infer_running: bool = False
    latest_frame_rgb: np.ndarray | None = None
    last_status_text: str = "Dung"
    last_infer_time: float = 0.0
    min_infer_interval_sec: float = 0.11

    def _ensure_backend_started(self) -> None:
        if not self.backend_infer_running:
            if safe_api_post("/infer/start"):
                self.backend_infer_running = True

    def _ensure_backend_stopped(self) -> None:
        if self.backend_infer_running:
            if safe_api_post("/infer/stop"):
                self.backend_infer_running = False

    def fetch_local_settings(self) -> None:
        data = safe_api_get("/settings")
        if not data:
            return
        self.unknown_debounce_sec = int(data.get("unknown_debounce_sec", 5))
        self.known_debounce_min = int(data.get("known_debounce_min", 1))

    def start(self, source: str) -> str:
        self.stop()
        self.camera_source = str(source or "0")

        if RUNNING_IN_DOCKER and self.camera_source.isdigit():
            self.running = True
            self.use_browser_webcam = True
            self.fetch_local_settings()
            self._ensure_backend_started()
            self.last_status_text = "Dang chay bang browser webcam mode (full Docker)"
            return self.last_status_text

        src: int | str = int(self.camera_source) if self.camera_source.isdigit() else self.camera_source
        self.use_browser_webcam = False

        if isinstance(src, str) and src.startswith("rtsp"):
            import os

            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap or not self.cap.isOpened():
            self.cap = None
            self.running = False
            return "Khong mo duoc camera/video source"

        self.running = True
        self.fetch_local_settings()
        self._ensure_backend_started()
        self.last_status_text = "Dang chay"
        return self.last_status_text

    def stop(self) -> None:
        self.running = False
        self.use_browser_webcam = False
        self._ensure_backend_stopped()
        self.latest_frame_rgb = None
        self.last_status_text = "Da dung"
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _process_inference_frame(self, frame_rgb: np.ndarray, source_label: str) -> tuple[np.ndarray, str, str, str]:
        # Gradio webcam frames can be readonly; OpenCV drawing APIs require writable arrays.
        frame_rgb = np.ascontiguousarray(frame_rgb.copy())

        now = time.time()
        dt = max(now - self.last_tick, 1e-6)
        fps = 1.0 / dt
        self.last_tick = now

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = infer_faces(frame_bgr)

        for face in results:
            bbox = face.get("bbox", [0, 0, 0, 0])
            name = face.get("name", "Unknown")
            similarity = float(face.get("similarity", 0.0))

            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_rgb,
                f"{name} {similarity:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            face_img = crop_face(frame_rgb, bbox)
            if face_img.size == 0:
                continue

            face_bytes = encode_face(face_img)
            if face_bytes is None:
                continue

            now_str = datetime.datetime.now().strftime("%H:%M:%S")

            if name == "Unknown":
                if now - self.last_unknown_seen > self.unknown_debounce_sec:
                    self.last_unknown_seen = now
                    b64_img = base64_encode(face_bytes)
                    self.unknown_items.insert(0, {"image": b64_img, "time": now_str})
                    self.unknown_items = self.unknown_items[:30]
                    log_unknown(face_bytes)
            else:
                last_seen = self.known_faces_history.get(name, 0)
                if now - last_seen > self.known_debounce_min * 60:
                    self.known_faces_history[name] = now
                    b64_img = base64_encode(face_bytes)
                    self.attendance_items.insert(
                        0,
                        {
                            "name": name,
                            "similarity": similarity,
                            "time": now_str,
                            "image": b64_img,
                        },
                    )
                    self.attendance_items = self.attendance_items[:100]
                    log_attendance(name, similarity, face_bytes)

        cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        self.latest_frame_rgb = frame_rgb.copy()

        status = f"Dang chay | FPS: {fps:.1f} | Source: {source_label}"
        if LAST_INFER_ERROR:
            status = f"{status} | InferErr: {LAST_INFER_ERROR}"
        self.last_status_text = status

        return (
            frame_rgb,
            render_attendance_html(self.attendance_items),
            render_unknown_html(self.unknown_items),
            status,
        )

    def process_browser_webcam_frame(self, frame_rgb: np.ndarray | None) -> tuple[np.ndarray, str, str, str]:
        frame_ready = coerce_frame_to_rgb_array(frame_rgb)

        if RUNNING_IN_DOCKER and frame_ready is not None and (not self.running or not self.use_browser_webcam):
            self.running = True
            self.use_browser_webcam = True
            self.fetch_local_settings()
            self._ensure_backend_started()

        if not self.running or not self.use_browser_webcam:
            return self.process_frame()

        if frame_ready is None:
            if self.latest_frame_rgb is not None:
                return (
                    self.latest_frame_rgb,
                    render_attendance_html(self.attendance_items),
                    render_unknown_html(self.unknown_items),
                    self.last_status_text,
                )

            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Dang cho webcam frame tu browser...",
                (30, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (220, 220, 220),
                2,
            )
            frame_out = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            self.last_status_text = "Dang cho webcam frame tu browser..."
            return (
                frame_out,
                render_attendance_html(self.attendance_items),
                render_unknown_html(self.unknown_items),
                self.last_status_text,
            )

        now = time.time()
        if now - self.last_infer_time >= self.min_infer_interval_sec:
            self.last_infer_time = now
            return self._process_inference_frame(frame_ready, "browser-webcam")

        # Keep stream visually continuous between inference frames.
        frame_out = np.ascontiguousarray(frame_ready.copy())
        dt = max(now - self.last_tick, 1e-6)
        fps = 1.0 / dt
        self.last_tick = now
        cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        self.latest_frame_rgb = frame_out.copy()
        self.last_status_text = f"Dang chay | FPS: {fps:.1f} | Source: browser-webcam"
        return (
            frame_out,
            render_attendance_html(self.attendance_items),
            render_unknown_html(self.unknown_items),
            self.last_status_text,
        )

    def process_frame(self) -> tuple[np.ndarray, str, str, str]:
        if self.running and self.use_browser_webcam:
            if self.latest_frame_rgb is not None:
                return (
                    self.latest_frame_rgb,
                    render_attendance_html(self.attendance_items),
                    render_unknown_html(self.unknown_items),
                    self.last_status_text,
                )

            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Browser webcam mode active",
                (40, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (220, 220, 220),
                2,
            )
            cv2.putText(
                placeholder,
                "Allow webcam and keep camera panel on",
                (25, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (190, 190, 190),
                1,
            )
            frame_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            self.last_status_text = "Dang cho webcam frame tu browser..."
            return frame_rgb, render_attendance_html(self.attendance_items), render_unknown_html(self.unknown_items), self.last_status_text

        if not self.running or self.cap is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Stopped - Press Start",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (220, 220, 220),
                2,
            )
            frame_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            self.last_status_text = "Dung"
            return frame_rgb, render_attendance_html(self.attendance_items), render_unknown_html(self.unknown_items), self.last_status_text

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            self.stop()
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "End of stream", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
            frame_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            self.last_status_text = "Het luong video"
            return frame_rgb, render_attendance_html(self.attendance_items), render_unknown_html(self.unknown_items), self.last_status_text

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._process_inference_frame(frame_rgb, self.camera_source)


def base64_encode(image_bytes: bytes) -> str:
    import base64

    return base64.b64encode(image_bytes).decode("ascii")


def get_models_and_settings() -> tuple[list[str], list[str], dict[str, Any]]:
    models = safe_api_get("/models") or {"models": []}
    settings = safe_api_get("/settings") or {}

    model_list = models.get("models", [])
    det_choices = [m for m in model_list if "det" in m or "yolo" in m or "scrfd" in m] or model_list
    rec_choices = [m for m in model_list if "w600k" in m or "arcface" in m or "glint" in m] or model_list
    return det_choices, rec_choices, settings


def refresh_settings_ui() -> tuple[Any, Any, float, float, int, int, str]:
    det_choices, rec_choices, settings = get_models_and_settings()

    det_value = (settings.get("det_weight", "").split("/")[-1] if settings else "")
    rec_value = (settings.get("rec_weight", "").split("/")[-1] if settings else "")
    sim = float(settings.get("similarity_thresh", 0.4))
    conf = float(settings.get("confidence_thresh", 0.5))
    unknown_sec = int(settings.get("unknown_debounce_sec", 5))
    known_min = int(settings.get("known_debounce_min", 1))

    return (
        gr.update(choices=det_choices, value=det_value if det_value in det_choices else None),
        gr.update(choices=rec_choices, value=rec_value if rec_value in rec_choices else None),
        sim,
        conf,
        unknown_sec,
        known_min,
        "Da tai setting tu backend",
    )


def apply_settings(
    det_model: str | None,
    rec_model: str | None,
    sim_thresh: float,
    conf_thresh: float,
    unknown_sec: int,
    known_min: int,
    state: PipelineState,
) -> tuple[str, PipelineState]:
    payload = {
        "det_weight": f"./weights/{det_model}" if det_model else None,
        "rec_weight": f"./weights/{rec_model}" if rec_model else None,
        "similarity_thresh": float(sim_thresh),
        "confidence_thresh": float(conf_thresh),
        "unknown_debounce_sec": int(unknown_sec),
        "known_debounce_min": int(known_min),
    }

    if safe_api_post("/settings", json_payload=payload):
        state.fetch_local_settings()
        return "Da cap nhat setting", state
    return "Khong cap nhat duoc setting", state


def force_update_db() -> str:
    ok = safe_api_post("/database/update")
    return "Da force update database" if ok else "Force update database that bai"


def start_pipeline(source: str, state: PipelineState) -> tuple[str, PipelineState]:
    status = state.start(source)
    return status, state


def stop_pipeline(state: PipelineState) -> tuple[str, PipelineState]:
    state.stop()
    return "Da dung", state


def tick(state: PipelineState) -> tuple[np.ndarray, str, str, str, PipelineState]:
    if state.running and state.use_browser_webcam:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), state
    frame, attendance_html, unknown_html, status = state.process_frame()
    return frame, attendance_html, unknown_html, status, state


def make_test_frame(title: str, subtitle: str) -> np.ndarray:
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (35, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    cv2.putText(canvas, subtitle, (35, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def initialize_dashboard(state: PipelineState) -> tuple[np.ndarray, str, str, str, PipelineState]:
    frame, attendance_html, unknown_html, status = state.process_frame()
    return frame, attendance_html, unknown_html, status, state


def render_test_frame(state: PipelineState) -> tuple[np.ndarray, str, PipelineState]:
    frame = make_test_frame("UI render OK", "Neu ban thay anh nay, frontend da hien thi binh thuong")
    return frame, "UI render test: OK", state


def on_browser_webcam_stream(frame: np.ndarray | None, state: PipelineState) -> tuple[np.ndarray, str, str, str, PipelineState]:
    if frame is None:
        print("[frontend] webcam stream callback: frame=None", flush=True)
    else:
        print(f"[frontend] webcam stream callback: frame_shape={getattr(frame, 'shape', None)}", flush=True)
    frame_out, attendance_html, unknown_html, status = state.process_browser_webcam_frame(frame)
    return frame_out, attendance_html, unknown_html, status, state


CSS = """
.face-strip {
  display: flex;
  flex-direction: row;
  gap: 8px;
  overflow-x: auto;
  padding: 4px;
}
.face-card {
  min-width: 110px;
  border: 1px solid #d6d6d6;
  border-radius: 8px;
  padding: 6px;
  background: #ffffff;
  text-align: center;
}
.face-card.unknown .meta { color: #c0392b; font-weight: 600; }
.face-card img {
  width: 90px;
  height: 70px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid #ccc;
}
.att-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 560px;
  overflow-y: auto;
}
.att-card {
  display: grid;
  grid-template-columns: 60px 50px 1fr;
  gap: 8px;
  align-items: center;
  border: 1px solid #d6d6d6;
  border-radius: 8px;
  padding: 6px;
  background: #ffffff;
}
.att-card img.db, .att-card img.live {
  width: 100%;
  height: 56px;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid #ccc;
}
.att-card .name { font-weight: 700; }
.att-card .sim, .att-card .time, .face-card .meta span { color: #666; font-size: 12px; }
"""


ACTIVE_STATES: list[PipelineState] = []


def register_state(state: PipelineState) -> PipelineState:
    ACTIVE_STATES.append(state)
    return state


def release_all_cameras() -> None:
    for state in ACTIVE_STATES:
        try:
            state.stop()
        except Exception:
            pass


def on_session_closed() -> None:
    release_all_cameras()
    close_ws_client()


def cleanup() -> None:
    release_all_cameras()
    close_ws_client()


atexit.register(cleanup)


def build_ui() -> gr.Blocks:
    initial_state = PipelineState()
    initial_state.fetch_local_settings()
    register_state(initial_state)

    with gr.Blocks(title="FaceID Browser UI") as demo:
        gr.Markdown("## FaceID Browser Dashboard")
        state = gr.State(initial_state)

        with gr.Row():
            source = gr.Textbox(value="0", label="Video Source", placeholder="0, 1, file path, rtsp://...")
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            test_frame_btn = gr.Button("Render Test Frame")
            status_box = gr.Textbox(value="Dung", label="Status", interactive=False)

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                det_model = gr.Dropdown(choices=[], label="Detection Model")
                rec_model = gr.Dropdown(choices=[], label="Recognition Model")
            with gr.Row():
                sim_thresh = gr.Slider(0.1, 1.0, value=0.4, step=0.01, label="Similarity Threshold")
                conf_thresh = gr.Slider(0.1, 1.0, value=0.5, step=0.01, label="Confidence Threshold")
            with gr.Row():
                unknown_sec = gr.Number(value=5, precision=0, label="Unknown Debounce (sec)")
                known_min = gr.Number(value=1, precision=0, label="Known Debounce (min)")
            with gr.Row():
                refresh_btn = gr.Button("Refresh Settings")
                apply_btn = gr.Button("Apply Settings", variant="primary")
                update_db_btn = gr.Button("Force Update DB")
            settings_msg = gr.Textbox(label="Settings Message", interactive=False)

        webcam_input = gr.Image(
            label="Browser Webcam Input (click camera icon and allow permission)",
            type="numpy",
            sources=["webcam"],
            streaming=True,
            interactive=True,
            webcam_options=gr.WebcamOptions(mirror=False),
        )

        frame_view = gr.Image(
            label="Live Stream",
            type="numpy",
            format="jpeg",
            interactive=False,
            value=make_test_frame("FaceID dashboard ready", "Bam Start de chay pipeline hoac Render Test Frame de kiem tra UI"),
        )

        with gr.Row():
            unknown_html = gr.HTML(label="Unknown List", value="<div class='face-strip'></div>")
            attendance_html = gr.HTML(label="Attendance List", value="<div class='att-list'></div>")

        timer = gr.Timer(value=0.12, active=True)

        demo.load(
            fn=refresh_settings_ui,
            inputs=None,
            outputs=[det_model, rec_model, sim_thresh, conf_thresh, unknown_sec, known_min, settings_msg],
        )

        demo.load(
            fn=initialize_dashboard,
            inputs=[state],
            outputs=[frame_view, attendance_html, unknown_html, status_box, state],
        )

        refresh_btn.click(
            fn=refresh_settings_ui,
            inputs=None,
            outputs=[det_model, rec_model, sim_thresh, conf_thresh, unknown_sec, known_min, settings_msg],
        )

        apply_btn.click(
            fn=apply_settings,
            inputs=[det_model, rec_model, sim_thresh, conf_thresh, unknown_sec, known_min, state],
            outputs=[settings_msg, state],
        )

        update_db_btn.click(fn=force_update_db, inputs=None, outputs=settings_msg)

        start_btn.click(fn=start_pipeline, inputs=[source, state], outputs=[status_box, state])
        stop_btn.click(fn=stop_pipeline, inputs=[state], outputs=[status_box, state])

        test_frame_btn.click(
            fn=render_test_frame,
            inputs=[state],
            outputs=[frame_view, status_box, state],
            queue=False,
        )

        timer.tick(
            fn=tick,
            inputs=[state],
            outputs=[frame_view, attendance_html, unknown_html, status_box, state],
            queue=False,
        )

        webcam_input.stream(
            fn=on_browser_webcam_stream,
            inputs=[webcam_input, state],
            outputs=[frame_view, attendance_html, unknown_html, status_box, state],
            queue=False,
            trigger_mode="always_last",
            stream_every=0.11,
        )

        demo.unload(on_session_closed)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        show_error=True,
        css=CSS,
    )