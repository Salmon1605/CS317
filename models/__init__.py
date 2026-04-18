from pathlib import Path
from typing import Protocol

import numpy as np

from .arcface import ArcFace
from .scrfd import SCRFD
from .yolo_face import YOLOFace


class FaceDetector(Protocol):
    conf_thres: float

    def detect(
        self,
        image: np.ndarray,
        max_num: int = 0,
        metric: str = "max",
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


def create_detector(
    model_path: str,
    input_size: tuple[int, int] = (640, 640),
    conf_thres: float = 0.5,
    iou_thres: float = 0.4,
) -> FaceDetector:
    model_name = Path(model_path).name.lower()
    if "yolo" in model_name and "face" in model_name:
        return YOLOFace(
            model_path=model_path,
            input_size=input_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
    return SCRFD(
        model_path=model_path,
        input_size=input_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )


__all__ = ["ArcFace", "SCRFD", "YOLOFace", "FaceDetector", "create_detector"]