import threading
from pathlib import Path

import cv2
import numpy as np

_model_cache = {}
_model_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize inference calls (defensive; main caller is already single-threaded)

def get_model(model_path: Path):
    from ultralytics import YOLO
    key = str(model_path)
    if key not in _model_cache:
        with _model_lock:
            if key not in _model_cache:
                _model_cache[key] = YOLO(key)
    return _model_cache[key]


def crop_to_detection(image_path: Path, output_path: Path, model_path: Path, conf: float = 0.25, imgsz: int = 640, masked: bool = False) -> bool:
    """Crop image_path to the highest-confidence detection and write it to output_path.

    The source image is never modified. Returns False (and writes nothing) when
    nothing is detected.

    masked=False (default): crop to the detection bounding box only.
    masked=True: apply the segmentation polygon — zero out pixels outside the
    mask and crop to the tight bounding box of the polygon. Use this when the
    ROI shape is non-rectangular and surrounding tissue should be excluded."""
    model = get_model(model_path)
    with _infer_lock:
        results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)

    if not results:
        return False

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False

    best_idx = int(boxes.conf.argmax())
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]

    if masked:
        segs = results[0].masks
        if segs is None or len(segs) == 0:
            return False

        # xy gives polygon coords already scaled to original image size
        poly = segs.xy[best_idx].astype(np.int32)
        if len(poly) < 3:
            return False

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        out = img.copy()
        out[mask == 0] = 0  # black background outside tongue region

        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return False

        crop = out[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return False

        crop = img[y1:y2, x1:x2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), crop))
