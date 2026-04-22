# detection.py — YOLOv8 inference and vehicle-class filtering.

import numpy as np
from ultralytics import YOLO
from config import VEHICLE_CLASSES


def count_vehicles(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = 0.35,
) -> tuple[list[dict], list[str]]:
    """
    Run YOLOv8 inference on a single frame and return only vehicle detections.

    Returns:
        detections  — list of dicts: {bbox, class_name, conf, weight}
        class_names — full class name list from the model
    """
    results = model.predict(frame, conf=conf_threshold, verbose=False, stream=False)

    detections: list[dict] = []
    class_names: dict = model.names  # {idx: name}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = class_names.get(cls_id, "").lower()
            weight   = VEHICLE_CLASSES.get(cls_name)
            if weight is None:
                continue  # skip non-vehicle classes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "class_name": cls_name,
                "conf":       float(box.conf[0]),
                "weight":     weight,
            })

    return detections, list(class_names.values())
