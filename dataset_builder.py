"""
dataset_builder.py
Interactive dataset builder for face recognition.

Usage:
    python dataset_builder.py

What it does:
- Prompts for a label (person name)
- Captures stable face crops into dataset/<label> folder
- Press SPACE to toggle auto/manual capture
- Press 's' in manual mode to save one sample
- Press 'q' to quit

This file includes a runnable main() guard so it executes when launched.
"""

import cv2
import os
import time

# ---------- Configuration ----------
DATA_DIR = "dataset"         # root folder for labeled images
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MIN_SIZE = (80, 80)         # consistent with detection script
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

# Number of images to capture per person by default (you can edit this)
TARGET_IMAGES = 60

# Temporal stability requirement (frames)
STABLE_FRAMES = 3

def ensure_label_folder(label):
    folder = os.path.join(DATA_DIR, label)
    os.makedirs(folder, exist_ok=True)
    return folder

def get_next_filename(folder, label, idx):
    ts = int(time.time() * 1000)
    return os.path.join(folder, f"{label}_{ts}_{idx}.jpg")

def build_dataset():
    print("dataset_builder starting...")
    label = input("Enter label/name (no spaces recommended, e.g., Alice): ").strip()
    if not label:
        print("Label empty; exiting.")
        return

    folder = ensure_label_folder(label)
    print(f"Saving up to {TARGET_IMAGES} images to: {folder}")
    print("Press 'q' to quit anytime. Press SPACE to toggle auto-capture mode (default: AUTO).")
    print("In MANUAL mode press 's' to save a sample.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam. Try changing camera index or check permissions.")
        return

    captured = 0
    auto_mode = True     # default: auto-capture when stable for STABLE_FRAMES
    stable_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: empty frame; retrying...")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(CASCADE_PATH).detectMultiScale(
                gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE
            )

            # If faces found, pick the largest one (assume user is main subject)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                stable_count += 1
            else:
                stable_count = 0

            cv2.putText(frame, f"Captured: {captured}/{TARGET_IMAGES}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            mode_text = "AUTO" if auto_mode else "MANUAL"
            cv2.putText(frame, f"Mode: {mode_text}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            cv2.imshow("Dataset Builder - Press SPACE to toggle mode, 's' to save in manual", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit requested by user ('q').")
                break
            if key == 32:  # SPACE toggles auto/manual
                auto_mode = not auto_mode
                print("Auto mode:", auto_mode)

            if auto_mode:
                if stable_count >= STABLE_FRAMES and len(faces) > 0:
                    x, y, w, h = faces[0]
                    crop = frame[y:y+h, x:x+w]
                    filename = get_next_filename(folder, label, captured)
                    cv2.imwrite(filename, crop)
                    captured += 1
                    stable_count = 0
                    print(f"[Saved AUTO] {filename} ({captured}/{TARGET_IMAGES})")
            else:
                # manual mode - press 's' to save current detection
                if key == ord('s') and len(faces) > 0:
                    x, y, w, h = faces[0]
                    crop = frame[y:y+h, x:x+w]
                    filename = get_next_filename(folder, label, captured)
                    cv2.imwrite(filename, crop)
                    captured += 1
                    print(f"[Saved MANUAL] {filename} ({captured}/{TARGET_IMAGES})")

            if captured >= TARGET_IMAGES:
                print("Target reached.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Dataset builder finished. Total captured:", captured)

if __name__ == "__main__":
    build_dataset()
