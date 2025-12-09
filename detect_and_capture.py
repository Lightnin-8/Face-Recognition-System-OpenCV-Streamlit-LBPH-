"""
detect_and_capture_stable.py
Improved version:
- Uses Haar cascade for detection (simple, fast)
- Applies Non-Maximum Suppression (NMS) to merge overlapping boxes
- Uses temporal confirmation: a face must be seen in N consecutive frames to be considered stable
- Saves face crops only for stable detections (reduces random false positives)
"""

import cv2
import os
import time
import numpy as np
from collections import deque, defaultdict

# ------------- Config -------------
OUTPUT_DIR = "captured_faces"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Detector parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (80, 80)  # slightly larger to reduce tiny false positives

# Stability / temporal params
STABLE_FRAMES_REQUIRED = 3  # must see this detection for 3 consecutive frames
MAX_MISSES = 1             # allowed misses before we forget the track

# NMS parameters
NMS_OVERLAP_THRESH = 0.4   # IOU threshold for merging overlapping boxes

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------- Helper functions -------------
def rect_to_xyxy(rect):
    x, y, w, h = rect
    return [x, y, x + w, y + h]

def xyxy_to_rect(xy):
    x1, y1, x2, y2 = xy
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

def non_max_suppression(boxes, scores=None, iou_threshold=0.4):
    """
    Simple NMS implementation.
    boxes: list of [x1,y1,x2,y2]
    scores: optional list of confidence scores (same length as boxes)
    Returns: list of selected boxes (as [x1,y1,x2,y2])
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # If no scores provided, sort by area (largest first) to keep bigger detections
    if scores is None:
        idxs = np.argsort(areas)[::-1]
    else:
        idxs = np.argsort(scores)[::-1]

    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter)

        # keep indices with IoU <= threshold
        keep_idxs = np.where(iou <= iou_threshold)[0]
        idxs = idxs[keep_idxs + 1]

    return boxes[pick].astype(int).tolist()

# ------------- Load detector & camera -------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Failed to load cascade from {CASCADE_PATH}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try camera index 1 or check permissions.")

print("Camera opened. Press 'c' to capture stable faces, 'q' to quit.")

# ------------- Tracking / stability bookkeeping -------------
# We'll track candidate boxes by their approximate center location.
# For a simple tracker we will assign detections to tracked items by IoU.
tracked = {}  # id -> dict(count, last_xyxy, misses)
next_track_id = 0

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union

# ------------- Main loop -------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Empty frame. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

        # Convert detections to [x1,y1,x2,y2]
        boxes_xyxy = [rect_to_xyxy(d) for d in detections]

        # Apply NMS to reduce overlapping duplicates
        boxes_xyxy = non_max_suppression(boxes_xyxy, iou_threshold=NMS_OVERLAP_THRESH)

        # ---------- Associate detections with existing tracked objects ----------
        used_track_ids = set()
        new_tracked = {}

        for box in boxes_xyxy:
            # Try to match this box to an existing tracked item via IoU
            best_id = None
            best_iou = 0.0
            for tid, info in tracked.items():
                last_box = info["last_xyxy"]
                score = iou_xyxy(box, last_box)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > 0.3 and best_id is not None:
                # Match found: update that track
                info = tracked[best_id]
                info["count"] += 1
                info["last_xyxy"] = box
                info["misses"] = 0
                new_tracked[best_id] = info
                used_track_ids.add(best_id)
            else:
                # No good match: create a new track
                new_tracked[next_track_id] = {"count": 1, "last_xyxy": box, "misses": 0}
                next_track_id += 1

        # Increase misses for tracks not matched this frame
        for tid, info in tracked.items():
            if tid not in used_track_ids:
                info["misses"] += 1
                if info["misses"] <= MAX_MISSES:
                    # carry forward if within allowed misses
                    new_tracked[tid] = info
                # else we forget it (drop)

        tracked = new_tracked  # replace with updated tracks

        # ---------- Draw boxes and check stability ----------
        for tid, info in list(tracked.items()):
            x1, y1, x2, y2 = info["last_xyxy"]
            # Draw a thicker box for stable tracks, thin for new ones
            if info["count"] >= STABLE_FRAMES_REQUIRED:
                color = (0, 200, 0)  # stable = greener
                thickness = 3
                label = f"ID {tid} (stable)"
            else:
                color = (0, 150, 255)  # unstable = orange-ish
                thickness = 1
                label = f"ID {tid} (#{info['count']})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # If stable and user pressed 'c', we will save crops below
            # (Saving is handled on key press to avoid writing every frame.)

        cv2.imshow("Face Detection (stable)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Save crops for stable tracks only
            timestamp = int(time.time() * 1000)
            for tid, info in tracked.items():
                if info["count"] >= STABLE_FRAMES_REQUIRED:
                    x1, y1, x2, y2 = info["last_xyxy"]
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
                    crop = frame[y1:y2, x1:x2]
                    filename = os.path.join(OUTPUT_DIR, f"stable_face_{timestamp}_{tid}.jpg")
                    cv2.imwrite(filename, crop)
                    print(f"[Saved stable] {filename}")

        elif key == ord('q'):
            print("Quitting.")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cleaned up.")
