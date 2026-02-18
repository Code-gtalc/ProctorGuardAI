# python
"""
ProctorGuard AI - Hybrid 2.0 (Mahalanobis + Geometric Gating)
Minimal preview UI while processing at native camera resolution.
"""
import cv2
import time
import csv
import numpy as np
from openvino import Core
from win32api import GetSystemMetrics

# ==========================
# CONFIGURATION
# ==========================

DEVICE = "CPU"
FRAME_WIDTH = 640
EYE_SIZE = 40

SMOOTHING = 0.7
LEARNING_FRAMES = 150

MAHALANOBIS_THRESHOLD = 3.0
OUTSIDE_CONFIRM_FRAMES = 3

HEAD_WEIGHT = 0.015
GEOMETRIC_MARGIN = 1.2
ADAPT_RATE = 0.01

LOG_FILE = "gaze_log.csv"

# ==========================
# MODEL LOADING
# ==========================

def load_models():
    ie = Core()
    return {
        "face": ie.compile_model(
            ie.read_model("intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"),
            DEVICE),
        "landmarks": ie.compile_model(
            ie.read_model("intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"),
            DEVICE),
        "head_pose": ie.compile_model(
            ie.read_model("intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"),
            DEVICE),
        "gaze": ie.compile_model(
            ie.read_model("intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"),
            DEVICE),
    }

# ==========================
# UTILITIES
# ==========================

def preprocess(img, shape):
    # shape: (N,C,H,W) or similar; keep width/height extraction consistent
    _, _, H, W = shape
    img = cv2.resize(img, (W, H))
    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    return img

def crop_square(img, center, size):
    """
    Crop a square of `size` centered at `center` from `img`.
    Pads with black when the crop goes outside image bounds.
    Returns a (size,size,3) uint8 image.
    """
    cx, cy = int(round(center[0])), int(round(center[1]))
    half = size // 2

    x1 = cx - half
    x2 = cx + half
    y1 = cy - half
    y2 = cy + half

    h, w = img.shape[:2]

    # Overlap region in source
    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(w, x2)
    sy2 = min(h, y2)

    # If no overlap, return black square
    if sx1 >= sx2 or sy1 >= sy2:
        return np.zeros((size, size, 3), dtype=img.dtype)

    out = np.zeros((size, size, 3), dtype=img.dtype)

    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    out[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]
    return cv2.resize(out, (size, size))

def largest_face(dets, frame_shape, conf=0.6):
    """
    Pick largest face box from face detection output.
    Returns (x1,y1,x2,y2) in full-frame integer coordinates or None.
    """
    H, W = frame_shape[:2]
    best = None
    # expected dets shape like [1,1,N,7]
    for det in dets[0][0]:
        score = float(det[2])
        if score < conf:
            continue
        x_min = int(det[3] * W)
        y_min = int(det[4] * H)
        x_max = int(det[5] * W)
        y_max = int(det[6] * H)

        # clamp
        x_min = max(0, min(W - 1, x_min))
        x_max = max(0, min(W, x_max))
        y_min = max(0, min(H - 1, y_min))
        y_max = max(0, min(H, y_max))

        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        if area == 0:
            continue
        if best is None or area > best[0]:
            best = (area, (x_min, y_min, x_max, y_max))
    return None if best is None else best[1]

# ==========================
# FEATURE EXTRACTION
# ==========================

def get_features(frame, models):
    """
    Process the input `frame` at its native resolution (no resizing here).
    Returns (dx, dy, yaw, pitch) or None when no face / bad crops.
    """
    H, W = frame.shape[:2]

    # Face detection on full-resolution frame
    face_blob = preprocess(frame, models["face"].input(0).shape)
    dets = list(models["face"](face_blob).values())[0]

    bbox = largest_face(dets, frame.shape)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    face = frame[y1:y2, x1:x2].copy()
    if face.size == 0:
        return None

    lm_blob = preprocess(face, models["landmarks"].input(0).shape)
    lm_out = list(models["landmarks"](lm_blob).values())[0].reshape(-1)

    fx, fy = face.shape[1], face.shape[0]
    lx = int(round(lm_out[0] * fx))
    ly = int(round(lm_out[1] * fy))
    rx = int(round(lm_out[2] * fx))
    ry = int(round(lm_out[3] * fy))

    left_eye = (x1 + lx, y1 + ly)
    right_eye = (x1 + rx, y1 + ry)

    le = crop_square(frame, left_eye, EYE_SIZE)
    re = crop_square(frame, right_eye, EYE_SIZE)
    if le is None or re is None:
        return None

    hp_blob = preprocess(face, models["head_pose"].input(0).shape)
    hp_out = models["head_pose"](hp_blob)
    yaw, pitch, roll = [v.flatten()[0] for v in hp_out.values()]

    gz_inputs = {
        models["gaze"].inputs[0].any_name: preprocess(le, models["gaze"].inputs[0].shape),
        models["gaze"].inputs[1].any_name: preprocess(re, models["gaze"].inputs[1].shape),
        models["gaze"].inputs[2].any_name: np.array([[yaw, pitch, roll]], dtype=np.float32)
    }

    gv = list(models["gaze"](gz_inputs).values())[0][0]

    dx = float(gv[0] + yaw * 0.002)
    dy = float(gv[1] + pitch * 0.002)

    return dx, dy, float(yaw), float(pitch)

# ==========================
# MAIN ENGINE
# ==========================

# python
def run():
    global FRAME_WIDTH, EYE_SIZE

    models = load_models()
    cap = cv2.VideoCapture(0)

    # Try request the monitor resolution (camera may ignore)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, GetSystemMetrics(0))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GetSystemMetrics(1))

    ret, frame = cap.read()
    if not ret:
        print("Unable to open camera")
        return

    # Use actual camera width/height for processing (no resizing in get_features)
    FRAME_WIDTH = frame.shape[1]
    EYE_SIZE = max(24, int(EYE_SIZE * (FRAME_WIDTH / 640)))

    prev_dx, prev_dy = 0.0, 0.0
    outside_counter = 0

    # Guided learning configuration: normalized screen points and per-point frame duration
    LEARNING_POINTS = [
        ("CENTER", 0.5, 0.5),
        ("LEFT",   0.20, 0.5),
        ("RIGHT",  0.80, 0.5),
        ("UP",     0.5, 0.20),
        ("DOWN",   0.5, 0.80),
    ]
    POINT_FRAMES = 40  # frames to collect per point (adjust if needed)

    learning_samples = []   # list of [dx, dy]
    learned = False
    point_idx = 0
    point_frame = 0

    log = open(LOG_FILE, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["timestamp", "status", "confidence"])

    # Minimal preview UI: create a small resizable window and set a small fixed size
    window_name = "ProctorGuard Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    PREVIEW_W, PREVIEW_H = 320, 240
    cv2.resizeWindow(window_name, PREVIEW_W, PREVIEW_H)

    try:
        frame_buffer = frame

        while True:
            if frame_buffer is not None:
                cur_frame = frame_buffer
                frame_buffer = None
            else:
                ret, cur_frame = cap.read()
                if not ret:
                    break

            H, W = cur_frame.shape[:2]

            feats = get_features(cur_frame, models)

            if feats is None:
                status = "NO_FACE"
                confidence = 0.0
            else:
                dx_raw, dy_raw, yaw, pitch = feats

                dx = SMOOTHING * prev_dx + (1 - SMOOTHING) * dx_raw
                dy = SMOOTHING * prev_dy + (1 - SMOOTHING) * dy_raw
                prev_dx, prev_dy = dx, dy

                if not learned:
                    # Draw guided target for the current learning point
                    name, nx, ny = LEARNING_POINTS[point_idx]
                    tx = int(round(nx * W))
                    ty = int(round(ny * H))

                    # overlay target marker and instruction
                    cv2.circle(cur_frame, (tx, ty), max(8, EYE_SIZE//3), (0, 255, 255), -1)
                    cv2.putText(cur_frame, f"Look at: {name}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    remaining = POINT_FRAMES - point_frame
                    cv2.putText(cur_frame, f"Hold for: {remaining}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

                    # collect gaze sample for this point
                    learning_samples.append([dx, dy])
                    point_frame += 1

                    # advance to next point when done
                    if point_frame >= POINT_FRAMES:
                        point_frame = 0
                        point_idx = (point_idx + 1) % len(LEARNING_POINTS)

                    status = "LEARNING"
                    confidence = 1.0

                    # finish learning when we have enough total frames
                    if len(learning_samples) >= LEARNING_FRAMES:
                        samples = np.array(learning_samples)
                        mean_gaze = np.mean(samples, axis=0)
                        cov_gaze = np.cov(samples.T)
                        inv_cov = np.linalg.inv(cov_gaze + 1e-6 * np.eye(2))

                        H_THRESHOLD = np.percentile(np.abs(samples[:, 0]), 95) * GEOMETRIC_MARGIN
                        V_THRESHOLD = np.percentile(np.abs(samples[:, 1]), 95) * GEOMETRIC_MARGIN

                        learned = True
                        print("Learning complete.")
                else:
                    diff = np.array([dx, dy]) - mean_gaze
                    maha = np.sqrt(diff.T @ inv_cov @ diff)
                    maha_score = max(0.0, 1.0 - maha / MAHALANOBIS_THRESHOLD)

                    horizontal_score = abs(dx) + HEAD_WEIGHT * abs(yaw)
                    vertical_score = abs(dy) + HEAD_WEIGHT * abs(pitch)

                    geometric_inside = (horizontal_score <= H_THRESHOLD and vertical_score <= V_THRESHOLD)

                    if geometric_inside:
                        final_score = maha_score
                    else:
                        final_score = 0.0

                    confidence = max(0.0, min(1.0, final_score))

                    if final_score > 0.3:
                        status = "INSIDE"
                        outside_counter = 0
                        mean_gaze = (1 - ADAPT_RATE) * mean_gaze + ADAPT_RATE * np.array([dx, dy])
                    else:
                        outside_counter += 1
                        if outside_counter >= OUTSIDE_CONFIRM_FRAMES:
                            status = "OUTSIDE"
                        else:
                            status = "INSIDE"

            writer.writerow([time.time(), status, round(float(confidence), 3)])
            log.flush()

            color = (0, 255, 0) if status == "INSIDE" else (0, 0, 255)
            cv2.putText(cur_frame, f"{status} | Conf:{confidence:.2f}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Show a small preview while processing full-resolution frames
            preview = cv2.resize(cur_frame, (PREVIEW_W, PREVIEW_H))
            cv2.imshow(window_name, preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
