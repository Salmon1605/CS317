import sys
import os
import cv2
import json
import time
import asyncio
import requests
import datetime
import threading
import traceback
import websockets
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFormLayout, QGroupBox, QScrollArea,
    QFrame, QMessageBox, QSplitter, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen

API_BASE_URL = "http://localhost:8000/api"
WS_URL = "ws://localhost:8000/ws/infer"


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Configuration")
        self.setMinimumWidth(400)
        
        self.layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.source_input = QLineEdit("0")
        form_layout.addRow("Video Source:", self.source_input)
        
        self.det_combo = QComboBox()
        self.rec_combo = QComboBox()
        form_layout.addRow("Detection Model:", self.det_combo)
        form_layout.addRow("Recognition Model:", self.rec_combo)
        
        self.sim_thresh_input = QLineEdit("0.4")
        form_layout.addRow("Similarity Threshold:", self.sim_thresh_input)
        
        self.conf_thresh_input = QLineEdit("0.5")
        form_layout.addRow("Confidence Threshold:", self.conf_thresh_input)
        
        self.unknown_debounce_sec_input = QLineEdit("5")
        form_layout.addRow("Unknown Debounce (sec):", self.unknown_debounce_sec_input)
        
        self.known_debounce_min_input = QLineEdit("5")
        form_layout.addRow("Known Debounce (min):", self.known_debounce_min_input)
        
        self.layout.addLayout(form_layout)
        
        # Action Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Update DB Button
        self.update_db_btn = QPushButton("Force Update DB")
        self.update_db_btn.clicked.connect(self.force_update_db)
        self.layout.addWidget(self.update_db_btn)
        
        self.layout.addWidget(button_box)
        
        self.load_models()
        self.load_settings()

    def load_models(self):
        try:
            r = requests.get(f"{API_BASE_URL}/models")
            if r.status_code == 200:
                models = r.json().get("models", [])
                self.det_combo.addItems([m for m in models if "det" in m or "yolo" in m or "scrfd" in m])
                self.rec_combo.addItems([m for m in models if "w600k" in m or "arcface" in m or "glint" in m])
                
                # Defaults if filtering fails
                if self.det_combo.count() == 0:
                    self.det_combo.addItems(models)
                if self.rec_combo.count() == 0:
                    self.rec_combo.addItems(models)
        except Exception as e:
            print(f"Failed to load models: {e}")

    def load_settings(self):
        try:
            r = requests.get(f"{API_BASE_URL}/settings")
            if r.status_code == 200:
                data = r.json()
                self.sim_thresh_input.setText(str(data.get("similarity_thresh", 0.4)))
                self.conf_thresh_input.setText(str(data.get("confidence_thresh", 0.5)))
                self.unknown_debounce_sec_input.setText(str(data.get("unknown_debounce_sec", 5)))
                self.known_debounce_min_input.setText(str(data.get("known_debounce_min", 5)))
                
                dv = data.get("det_weight", "").split("/")[-1]
                rv = data.get("rec_weight", "").split("/")[-1]
                
                if dv:
                    idx = self.det_combo.findText(dv)
                    if idx >= 0: self.det_combo.setCurrentIndex(idx)
                if rv:
                    idx = self.rec_combo.findText(rv)
                    if idx >= 0: self.rec_combo.setCurrentIndex(idx)
        except Exception as e:
            print(f"Failed to load settings: {e}")

    def force_update_db(self):
        try:
            r = requests.post(f"{API_BASE_URL}/database/update")
            if r.status_code == 200:
                QMessageBox.information(self, "Success", "Database updated successfully!")
            else:
                QMessageBox.warning(self, "Error", f"Failed to update DB: {r.text}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to connect: {e}")

    def apply_settings(self):
        payload = {
            "det_weight": f"./weights/{self.det_combo.currentText()}" if self.det_combo.currentText() else None,
            "rec_weight": f"./weights/{self.rec_combo.currentText()}" if self.rec_combo.currentText() else None,
            "confidence_thresh": float(self.conf_thresh_input.text() or 0.5),
            "similarity_thresh": float(self.sim_thresh_input.text() or 0.4),
            "unknown_debounce_sec": int(self.unknown_debounce_sec_input.text() or 5),
            "known_debounce_min": int(self.known_debounce_min_input.text() or 5),
        }
        try:
            requests.post(f"{API_BASE_URL}/settings", json=payload)
        except Exception as e:
            print(f"Error applying settings: {e}")
            
    def accept(self):
        self.apply_settings()
        super().accept()


class InferenceWorker(QThread):
    result_signal = pyqtSignal(np.ndarray, list, float)
    error_signal = pyqtSignal(str)

    def __init__(self, ws_url=WS_URL):
        super().__init__()
        self.ws_url = ws_url
        self.running = True
        self.frame_queue = []
        
    # Kích thước tối đa gửi lên API (giảm băng thông, tăng tốc độ)
    INFER_MAX_WIDTH = 1280

    def add_frame(self, frame):
        # Only keep the absolute latest frame to avoid latency lag on high-res RTSP
        self.frame_queue = [frame]

    def run(self):
        asyncio.run(self.ws_loop())
        
    async def ws_loop(self):
        last_time = time.time()
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.error_signal.emit("WS Connected")
                    while self.running:
                        if self.frame_queue:
                            frame = self.frame_queue.pop(0)
                            
                            # Cal FPS
                            curr_time = time.time()
                            fps = 1.0 / (curr_time - last_time) if (curr_time - last_time) > 0 else 0
                            last_time = curr_time
                            
                            # Resize frame trước khi gửi để giảm dung lượng
                            orig_h, orig_w = frame.shape[:2]
                            if orig_w > self.INFER_MAX_WIDTH:
                                scale = self.INFER_MAX_WIDTH / orig_w
                                infer_w = self.INFER_MAX_WIDTH
                                infer_h = round(orig_h * scale)
                                small_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                            else:
                                scale = 1.0
                                small_frame = frame
                            
                            # Encode JPEG quality 70 – đủ cho nhận diện khuôn mặt, nhỏ hơn nhiều
                            ret, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if not ret:
                                continue
                            
                            await ws.send(buffer.tobytes())
                            response = await ws.recv()
                            data = json.loads(response)
                            
                            if "error" in data:
                                self.error_signal.emit(data["error"])
                            elif "results" in data:
                                # Scale bbox về kích thước frame gốc nếu đã resize
                                if scale != 1.0:
                                    inv = 1.0 / scale
                                    for face in data["results"]:
                                        face["bbox"] = [int(v * inv) for v in face["bbox"]]
                                self.result_signal.emit(frame, data["results"], fps)
                        else:
                            await asyncio.sleep(0.01)
            except Exception as e:
                self.error_signal.emit(f"WS Error: {e}")
                await asyncio.sleep(1)

    def stop(self):
        self.running = False
        self.wait()


class CameraWorker(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, source="0"):
        super().__init__()
        if str(source).isdigit():
            self.source = int(source)
        else:
            self.source = source
        self.running = True
        self._cap = None
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._reader_stop = True
        
    def _reader_loop(self):
        while not self._reader_stop:
            try:
                if self._cap and self._cap.isOpened():
                    ret, frame = self._cap.read()
                    with self._lock:
                        self._ret = ret
                        self._frame = frame
                else:
                    time.sleep(0.1)
            except Exception:
                if self._reader_stop:
                    break
                time.sleep(0.1)

    def _read(self):
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def run(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._cap = cv2.VideoCapture(self.source)
            
        if not self._cap.isOpened():
            print("Camera failed to read frame.")
            return
            
        self._reader_stop = False
        t = threading.Thread(target=self._reader_loop, daemon=True)
        t.start()
        time.sleep(0.5)

        while self.running:
            ret, frame = self._read()
            if ret and frame is not None:
                self.new_frame_signal.emit(frame)
            time.sleep(1/30.0)

        self._reader_stop = True
        time.sleep(0.2)
        if self._cap:
            self._cap.release()
        
    def stop(self):
        self.running = False
        self.wait()


class FaceItemWidget(QWidget):
    def __init__(self, face_img_rgb, name, time_str, similarity=None, is_unknown=False):
        super().__init__()
        
        layout = QHBoxLayout(self) if not is_unknown else QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        if is_unknown:
            self.setFixedSize(100, 120)
            
            # Live Image
            h, w, c = face_img_rgb.shape
            qimg = QImage(face_img_rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img_lbl = QLabel()
            img_lbl.setPixmap(pixmap)
            img_lbl.setFixedSize(60, 60)
            img_lbl.setStyleSheet("border-radius: 5px; border: 1px solid #ccc;")
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(img_lbl)
            
            # Info
            info_layout = QVBoxLayout()
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-weight: bold; font-size: 10pt; color: red;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            time_lbl = QLabel(f"Time: {time_str}")
            time_lbl.setStyleSheet("font-size: 8pt; color: #666;")
            time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            info_layout.addWidget(name_lbl)
            info_layout.addWidget(time_lbl)
            layout.addLayout(info_layout)
        else:
            self.setFixedSize(300, 80)
            
            images_layout = QHBoxLayout()
            images_layout.setSpacing(5)
            
            # Reference DB Image
            ref_lbl = QLabel()
            ref_lbl.setFixedSize(60, 60)
            ref_lbl.setStyleSheet("border-radius: 5px; border: 1px solid #777;")
            try:
                # Load from API
                url = f"http://localhost:8000/faces/{name}.jpg"
                res = requests.get(url, timeout=1)
                if res.status_code == 200:
                    ref_qimg = QImage()
                    ref_qimg.loadFromData(res.content)
                    ref_pix = QPixmap.fromImage(ref_qimg).scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    ref_lbl.setPixmap(ref_pix)
                else:
                    ref_lbl.setText("No DB\nIMG")
                    ref_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            except:
                ref_lbl.setText("Error")
                ref_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
            images_layout.addWidget(ref_lbl)
            
            # Live Frame Image
            h, w, c = face_img_rgb.shape
            live_qimg = QImage(face_img_rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
            live_pix = QPixmap.fromImage(live_qimg).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            live_lbl = QLabel()
            live_lbl.setPixmap(live_pix)
            live_lbl.setFixedSize(40, 40)
            live_lbl.setStyleSheet("border-radius: 3px; border: 1px solid #ccc;")
            
            # Align live image to bottom of the reference image
            live_container = QVBoxLayout()
            live_container.addStretch()
            live_container.addWidget(live_lbl)
            images_layout.addLayout(live_container)
            
            layout.addLayout(images_layout)
            
            # Info
            info_layout = QVBoxLayout()
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-weight: bold; font-size: 11pt; color: #333;")
            time_lbl = QLabel(f"Time: {time_str}")
            time_lbl.setStyleSheet("font-size: 8pt; color: #666;")
            
            info_layout.addWidget(name_lbl)
            if similarity is not None:
                sim_lbl = QLabel(f"Sim: {similarity:.2f}")
                sim_lbl.setStyleSheet("font-size: 8pt; color: #666;")
                info_layout.addWidget(sim_lbl)
            info_layout.addWidget(time_lbl)
            info_layout.addStretch()
            
            layout.addLayout(info_layout)
            
        self.setStyleSheet("""
            FaceItemWidget {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            FaceItemWidget:hover {
                background-color: #f0f8ff;
                border-color: #87cefa;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceID Management System")
        self.setGeometry(100, 100, 1280, 800)
        
        self.camera_source = "0"
        self.known_faces_history = {} # name: last_seen timestamp
        self.last_unknown_seen = 0 # last_seen timestamp
        self.is_running = False
        
        self.setup_ui()
        
        self.unknown_debounce_sec = 5
        self.known_debounce_min = 1
        
        # Load settings specifically for local use in GUI debounce
        self.fetch_local_settings()
        
        self.infer_worker = InferenceWorker()
        self.infer_worker.result_signal.connect(self.on_inference_result)
        self.infer_worker.error_signal.connect(self.on_ws_error)
        self.infer_worker.start()
        
        self.camera_worker = None
        # App does not auto-start; user must press Start

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #385723; color: white;")
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        title = QLabel("FACE ID")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; letter-spacing: 2px;")
        
        _btn_style = """
            QPushButton {
                background-color: transparent; border: 1px solid white; color: white;
                padding: 5px 15px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: rgba(255,255,255,0.2); }
        """
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setStyleSheet(_btn_style)
        self.start_btn.clicked.connect(self.start_pipeline)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(_btn_style.replace(
            "background-color: transparent", "background-color: #c0392b"
        ))
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.stop_btn.setEnabled(False)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setStyleSheet(_btn_style)
        settings_btn.clicked.connect(self.open_settings)
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.start_btn)
        header_layout.addWidget(self.stop_btn)
        header_layout.addWidget(settings_btn)
        main_layout.addWidget(header)
        
        # Content Splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Left Panel (Video + Unknown)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: white;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.video_label, stretch=1)
        
        # Unknown List Horizontal
        unknown_group = QGroupBox("Unknown List")
        unknown_group.setFixedHeight(180)
        unknown_layout = QHBoxLayout(unknown_group)
        
        self.unknown_scroll = QScrollArea()
        self.unknown_scroll.setWidgetResizable(True)
        self.unknown_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.unknown_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.unknown_container = QWidget()
        self.unknown_container_layout = QHBoxLayout(self.unknown_container)
        self.unknown_container_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.unknown_scroll.setWidget(self.unknown_container)
        
        unknown_layout.addWidget(self.unknown_scroll)
        left_layout.addWidget(unknown_group)
        
        content_splitter.addWidget(left_panel)
        
        # Right Panel (Attendance)
        right_panel = QGroupBox("Attendance List")
        right_panel.setFixedWidth(340)
        right_panel.setStyleSheet("QGroupBox { font-weight: bold; }")
        right_layout = QVBoxLayout(right_panel)
        
        self.attendance_scroll = QScrollArea()
        self.attendance_scroll.setWidgetResizable(True)
        self.attendance_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        self.attendance_container = QWidget()
        self.attendance_container_layout = QVBoxLayout(self.attendance_container)
        self.attendance_container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.attendance_scroll.setWidget(self.attendance_container)
        
        right_layout.addWidget(self.attendance_scroll)
        content_splitter.addWidget(right_panel)
        
        # Set Splitter ratio
        content_splitter.setSizes([900, 280])

    def fetch_local_settings(self):
        try:
            r = requests.get(f"{API_BASE_URL}/settings")
            if r.status_code == 200:
                data = r.json()
                self.unknown_debounce_sec = data.get("unknown_debounce_sec", 5)
                self.known_debounce_min = data.get("known_debounce_min", 1)
        except:
            pass

    def start_pipeline(self):
        """Start camera + inference pipeline."""
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # Notify API
        try:
            requests.post(f"{API_BASE_URL}/infer/start", timeout=2)
        except Exception:
            pass
        # Start camera
        if self.camera_worker:
            self.camera_worker.stop()
        self.camera_worker = CameraWorker(self.camera_source)
        self.camera_worker.new_frame_signal.connect(self.on_new_frame)
        self.camera_worker.start()

    def stop_pipeline(self):
        """Stop camera + inference pipeline."""
        if not self.is_running:
            return
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # Notify API
        try:
            requests.post(f"{API_BASE_URL}/infer/stop", timeout=2)
        except Exception:
            pass
        # Stop camera
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        # Clear video display
        self.video_label.clear()
        self.video_label.setStyleSheet("background-color: #111; color: #aaa;")
        self.video_label.setText("⏹  Stopped — press  ▶ Start  to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def start_camera(self):
        """Legacy helper — delegates to start_pipeline."""
        self.start_pipeline()

    def on_new_frame(self, frame):
        # We send it to backend
        self.infer_worker.add_frame(frame)

    def on_ws_error(self, err):
        # Print warning to console instead of showing on UI
        print(f"WS Status: {err}")
        
    def crop_face(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]

    def on_inference_result(self, frame_bgr, results, fps):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Draw FPS
        cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw Results
        for face in results:
            bbox = face["bbox"]
            name = face["name"]
            sim = face.get("similarity", 0.0)
            
            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
            
            # Draw bbox natively with opencv on the copy
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_rgb, f"{name} {sim:.2f}", (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            time_str = datetime.datetime.now().strftime("%H:%M:%S")
            current_time = time.time()
            
            # Add to lists
            if name == "Unknown":
                if current_time - self.last_unknown_seen > self.unknown_debounce_sec:
                    face_img = self.crop_face(frame_rgb, bbox)
                    if face_img.size > 0:
                        self.last_unknown_seen = current_time
                        widget = FaceItemWidget(face_img, "Unknown", time_str, is_unknown=True)
                        self.unknown_container_layout.insertWidget(0, widget)
                        # Encode face NOW (main thread) then log in background
                        face_bytes = self._encode_face(face_img)
                        threading.Thread(target=self._log_unknown, args=(face_bytes,), daemon=True).start()
                        # Cleanup old
                        if self.unknown_container_layout.count() > 30:
                            item = self.unknown_container_layout.takeAt(self.unknown_container_layout.count() - 1)
                            if item and item.widget():
                                item.widget().deleteLater()
            else:
                last_seen = self.known_faces_history.get(name, 0)
                if current_time - last_seen > (self.known_debounce_min * 60):
                    self.known_faces_history[name] = current_time
                    face_img = self.crop_face(frame_rgb, bbox)
                    if face_img.size > 0:
                        widget = FaceItemWidget(face_img, name, time_str, similarity=sim, is_unknown=False)
                        self.attendance_container_layout.insertWidget(0, widget)
                        # Encode face NOW (main thread) then log in background
                        face_bytes = self._encode_face(face_img)
                        threading.Thread(target=self._log_attendance, args=(name, sim, face_bytes), daemon=True).start()

        # Show frame
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    @staticmethod
    def _encode_face(face_img_rgb: np.ndarray) -> bytes | None:
        """Encode RGB face crop to JPEG bytes (convert to BGR first)."""
        try:
            face_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
            ret, buf = cv2.imencode('.jpg', face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return buf.tobytes() if ret else None
        except Exception:
            return None

    def _log_attendance(self, name: str, similarity: float, face_bytes: bytes | None = None):
        try:
            files = {'image': ('face.jpg', face_bytes, 'image/jpeg')} if face_bytes else {}
            data = {'name': name, 'similarity': str(float(similarity))}
            requests.post(f"{API_BASE_URL}/attendance/log", data=data, files=files, timeout=5)
        except Exception:
            pass

    def _log_unknown(self, face_bytes: bytes | None = None):
        try:
            files = {'image': ('face.jpg', face_bytes, 'image/jpeg')} if face_bytes else {}
            requests.post(f"{API_BASE_URL}/unknown/log", files=files, timeout=5)
        except Exception:
            pass

    def open_settings(self):
        self.fetch_local_settings()
        dlg = SettingsDialog(self)
        dlg.source_input.setText(str(self.camera_source))
        if dlg.exec():
            self.fetch_local_settings()
            new_source = dlg.source_input.text()
            if new_source != str(self.camera_source):
                self.camera_source = new_source
                # Only restart the camera if pipeline is currently running
                if self.is_running:
                    if self.camera_worker:
                        self.camera_worker.stop()
                    self.camera_worker = CameraWorker(self.camera_source)
                    self.camera_worker.new_frame_signal.connect(self.on_new_frame)
                    self.camera_worker.start()

    def closeEvent(self, event):
        if self.is_running:
            try:
                requests.post(f"{API_BASE_URL}/infer/stop", timeout=1)
            except Exception:
                pass
        self.infer_worker.stop()
        if self.camera_worker:
            self.camera_worker.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())