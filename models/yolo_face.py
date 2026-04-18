import logging
from typing import Tuple

import cv2
import numpy as np
import onnxruntime

__all__ = ["YOLOFace"]

logger = logging.getLogger(__name__)


class YOLOFace:
    """YOLO-based face detector with 5-point facial landmarks."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str, providers: list = None) -> None:
        """Initialize the ONNX inference session."""
        if providers is None:
            available = onnxruntime.get_available_providers()
            logger.info(f"YOLOFace onnxruntime module: {onnxruntime.__file__}")
            logger.info(f"YOLOFace available providers: {available}")
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append(
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kSameAsRequested",
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                        },
                    )
                )
            providers.append("CPUExecutionProvider")
            if "CUDAExecutionProvider" not in available:
                logger.warning("CUDAExecutionProvider is not available for YOLOFace. Inference will use CPU.")

        try:
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options=opts,
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"Successfully loaded YOLOFace model from {model_path}")
            logger.info(f"YOLOFace active providers: {self.session.get_providers()}")
        except Exception as e:
            logger.warning(f"Failed to load the model with optimal providers: {e}. Falling back to CPU...")
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"YOLOFace active providers: {self.session.get_providers()}")

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize with aspect-ratio preservation and pad to model input size."""
        input_w, input_h = self.input_size
        height, width = image.shape[:2]
        scale = min(input_w / width, input_h / height)

        resized_w = int(round(width * scale))
        resized_h = int(round(height * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        return canvas, scale, (pad_x, pad_y)

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        det_image, scale, pad = self._letterbox(image)
        rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        return blob, scale, pad

    def detect(
        self,
        image: np.ndarray,
        max_num: int = 0,
        metric: str = "max",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect faces and return SCRFD-compatible bboxes and landmarks."""
        blob, scale, (pad_x, pad_y) = self._preprocess(image)
        raw = self.session.run([self.output_name], {self.input_name: blob})[0]

        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] < 21:
            raise ValueError(f"Unexpected YOLOFace output shape: {raw.shape}")

        rows = raw[0]
        rows = rows[rows[:, 4] >= self.conf_thres]
        if rows.size == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        boxes = rows[:, :4].copy()
        scores = rows[:, 4].copy()
        kpss = rows[:, 6:21].reshape(-1, 5, 3).copy()

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        kpss[:, :, 0] = (kpss[:, :, 0] - pad_x) / scale
        kpss[:, :, 1] = (kpss[:, :, 1] - pad_y) / scale

        image_h, image_w = image.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_h - 1)
        kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, image_w - 1)
        kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, image_h - 1)

        det = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = self.nms(det, iou_thres=self.iou_thres)
        det = det[keep]
        kpss = kpss[keep, :, :2].astype(np.float32, copy=False)

        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - image_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area if metric == "max" else (area - offset_dist_squared * 2.0)
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex]
            kpss = kpss[bindex]

        return det, kpss

    def nms(self, dets: np.ndarray, iou_thres: float) -> list[int]:
        """Greedy non-maximum suppression."""
        if dets.size == 0:
            return []

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]

        return keep