# zones.py — Frame zone splitting and detection-to-zone assignment.

import numpy as np
from config import NUM_SIGNALS


def split_into_zones(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Divide a frame into NUM_SIGNALS equal vertical strips.
    Returns a list of (x1, y1, x2, y2) tuples, one per zone.
    """
    h, w = frame.shape[:2]
    zone_w = w // NUM_SIGNALS
    zones = []
    for i in range(NUM_SIGNALS):
        x1 = i * zone_w
        x2 = x1 + zone_w if i < NUM_SIGNALS - 1 else w  # last zone absorbs remainder
        zones.append((x1, 0, x2, h))
    return zones


def assign_to_zones(
    detections: list[dict],
    zones: list[tuple[int, int, int, int]],
) -> list[list[dict]]:
    """
    Map each detection to the zone whose x-range contains the bbox centre-x.
    Returns zone_detections[i] = list of detections belonging to zone i.
    """
    zone_detections: list[list[dict]] = [[] for _ in range(NUM_SIGNALS)]
    for det in detections:
        x1, _, x2, _ = det["bbox"]
        cx = (x1 + x2) // 2
        for i, (zx1, _, zx2, _) in enumerate(zones):
            if zx1 <= cx < zx2:
                zone_detections[i].append(det)
                break
    return zone_detections
