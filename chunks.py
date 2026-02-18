"""
ProctorGuard AI - Hybrid 2.0 (Mahalanobis + Geometric Gating)
Modified: interactive calibration (press a Start button per step) and head-pose tracking during calibration.
"""

import cv2
import time
import csv
import numpy as np
from openvino import Core

# ==========================
# CONFIGURATION
# ==========================

DEVICE = "CPU"
FRAME_WIDTH = 640
EYE_SIZE = 40

SMOOTHING = 0.7

# Learning / calibration
CALIBRATION_STEPS = ["CENTER", "LEFT", "RIGHT", "TOP", "BOTTOM"]
FRAMES_PER_STEP = 60
WAIT_BEFORE_CAPTURE = 2.0  # seconds to wait after pressing Start
ADAPT_RATE = 0.01

MAHALANOBIS_THRESHOLD = 3.0
OUTSIDE_CONFIRM_FRAMES = 3

HEAD_WEIGHT = 0.015  # head pose geometric contribution (kept as fusion weight)
GEOMETRIC_MARGIN = 1.2  # expand learned percentile by 20%

LOG_FILE = "gaze_log.csv"

# UI colors (BGR)
PROMPT_COLOR = (255, 0, 0)     # contrasting blue for prompts/countdown (replaces yellow)
PROGRESS_COLOR = (255, 0, 0)
BUTTON_COLOR = (50, 200, 50)
NO_FACE_COLOR = (0, 0, 255)
INSIDE_COLOR = (0, 255, 0)

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
    _, _, H, W = shape
    img = cv2.resize(img,(W,H))
    img = img.transpose(2,0,1)[None].astype(np.float32)
    return img

def crop_square(img, center, size):
    x,y = center
    s=size//2
    x1,x2=max(0,x-s),min(img.shape[1],x+s)
    y1,y2=max(0,y-s),min(img.shape[0],y+s)
    roi=img[y1:y2,x1:x2]
    if roi.size == 0:
        return np.zeros((size,size,3),dtype=np.uint8)
    return cv2.resize(roi,(size,size))

def largest_face(dets, shape, conf=0.6):
    H,W=shape[:2]
    best=None
    for det in dets[0][0]:
        if det[2]<conf: continue
        x1,y1,x2,y2=(det[3:]*[W,H,W,H]).astype(int)
        area=(x2-x1)*(y2-y1)
        if best is None or area>best[0]:
            best=(area,(x1,y1,x2,y2))
    return None if best is None else best[1]

# ==========================
# FEATURE EXTRACTION
# ==========================

def get_features(frame, models):

    scale=FRAME_WIDTH/frame.shape[1]
    frame=cv2.resize(frame,(FRAME_WIDTH,int(frame.shape[0]*scale)))

    face_blob=preprocess(frame,models["face"].input(0).shape)
    dets=list(models["face"](face_blob).values())[0]

    bbox=largest_face(dets,frame.shape)
    if bbox is None:
        return None

    x1,y1,x2,y2=bbox
    face=frame[y1:y2,x1:x2]

    lm_blob=preprocess(face,models["landmarks"].input(0).shape)
    lm=list(models["landmarks"](lm_blob).values())[0].reshape(-1)

    fx,fy=face.shape[1],face.shape[0]
    left_eye=(x1+int(lm[0]*fx),y1+int(lm[1]*fy))
    right_eye=(x1+int(lm[2]*fx),y1+int(lm[3]*fy))

    le=crop_square(frame,left_eye,EYE_SIZE)
    re=crop_square(frame,right_eye,EYE_SIZE)

    hp_blob=preprocess(face,models["head_pose"].input(0).shape)
    hp_out=models["head_pose"](hp_blob)
    yaw,pitch,roll=[v.flatten()[0] for v in hp_out.values()]

    gz_inputs={
        models["gaze"].inputs[0].any_name:
            preprocess(le,models["gaze"].inputs[0].shape),
        models["gaze"].inputs[1].any_name:
            preprocess(re,models["gaze"].inputs[1].shape),
        models["gaze"].inputs[2].any_name:
            np.array([[yaw,pitch,roll]],dtype=np.float32)
    }

    gv=list(models["gaze"](gz_inputs).values())[0][0]

    dx=gv[0]+yaw*0.002
    dy=gv[1]+pitch*0.002

    return float(dx),float(dy),float(yaw),float(pitch)

# ==========================
# MAIN ENGINE
# ==========================

def run():

    models=load_models()
    cap=cv2.VideoCapture(0)

    prev_dx,prev_dy=0,0
    outside_counter=0

    learning_samples=[]  # each entry: [dx,dy,yaw,pitch,step_label_index]
    learned=False

    mean_gaze=None
    inv_cov=None
    H_THRESHOLD=0.0
    V_THRESHOLD=0.0

    # UI / calibration state
    calib_index = 0
    waiting_for_start = False
    start_pressed = False
    start_time = None
    capturing = False
    frames_captured = 0

    # define button rect in window coords
    BTN_W, BTN_H = 180, 50
    BTN_X, BTN_Y = 30, 80
    button_rect = (BTN_X, BTN_Y, BTN_W, BTN_H)
    button_clicked = False

    # mouse callback for button
    def on_mouse(event, x, y, flags, param):
        nonlocal button_clicked
        bx,by,bw,bh = button_rect
        if event==cv2.EVENT_LBUTTONDOWN:
            if bx <= x <= bx+bw and by <= y <= by+bh:
                button_clicked = True

    cv2.namedWindow("ProctorGuard Hybrid 2.0")
    cv2.setMouseCallback("ProctorGuard Hybrid 2.0", on_mouse)

    log=open(LOG_FILE,"w",newline="")
    writer=csv.writer(log)
    writer.writerow(["timestamp","status","confidence"])

    while True:

        ret,frame=cap.read()
        if not ret:
            break

        feats=get_features(frame,models)

        status="NO_FACE"
        confidence=0.0

        # --- Calibration phase (interactive) ---
        if not learned:
            current_step = CALIBRATION_STEPS[calib_index] if calib_index < len(CALIBRATION_STEPS) else None

            if current_step is not None:
                # Show prompt and Start button (prompt color changed)
                cv2.putText(frame,f"Calibration: Look at {current_step}",(30,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,PROMPT_COLOR,2)

                # draw button
                bx,by,bw,bh = button_rect
                cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),BUTTON_COLOR,-1)
                cv2.putText(frame,"START",(bx+20,by+35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

                if button_clicked and not start_pressed and not capturing:
                    start_pressed = True
                    start_time = time.time()
                    button_clicked = False

                # if start pressed, show countdown wait
                if start_pressed and not capturing:
                    elapsed = time.time() - start_time
                    remaining = max(0.0, WAIT_BEFORE_CAPTURE - elapsed)
                    cv2.putText(frame,f"Starting in {remaining:.1f}s",(30,140),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,PROMPT_COLOR,2)
                    if elapsed >= WAIT_BEFORE_CAPTURE:
                        capturing = True
                        frames_captured = 0

                # capturing frames for this calibration step
                if capturing and feats is not None:
                    dx_raw,dy_raw,yaw,pitch = feats

                    dx=SMOOTHING*prev_dx+(1-SMOOTHING)*dx_raw
                    dy=SMOOTHING*prev_dy+(1-SMOOTHING)*dy_raw
                    prev_dx,prev_dy = dx,dy

                    learning_samples.append([dx,dy,yaw,pitch,calib_index])
                    frames_captured += 1

                    cv2.putText(frame,f"Capturing {frames_captured}/{FRAMES_PER_STEP}",
                                (30,200),cv2.FONT_HERSHEY_SIMPLEX,0.8,INSIDE_COLOR,2)

                    if frames_captured >= FRAMES_PER_STEP:
                        # finish step
                        capturing = False
                        start_pressed = False
                        calib_index += 1
                else:
                    # not capturing - still show guidance if no face
                    if feats is None:
                        cv2.putText(frame,"No face detected - adjust position",(30,170),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,NO_FACE_COLOR,2)

            # if all steps done, finalize learning
            if calib_index >= len(CALIBRATION_STEPS):
                if len(learning_samples) > 0:
                    samples = np.array(learning_samples)
                    gaze_2d = samples[:,0:2]
                    mean_gaze = np.mean(gaze_2d,axis=0)
                    cov_gaze = np.cov(gaze_2d.T)
                    inv_cov = np.linalg.inv(cov_gaze + 1e-6*np.eye(2))

                    # compute geometric thresholds using head pose augmented scores
                    horizontal_scores = np.abs(samples[:,0]) + HEAD_WEIGHT*np.abs(samples[:,2])
                    vertical_scores = np.abs(samples[:,1]) + HEAD_WEIGHT*np.abs(samples[:,3])

                    H_THRESHOLD = np.percentile(horizontal_scores,95) * GEOMETRIC_MARGIN
                    V_THRESHOLD = np.percentile(vertical_scores,95) * GEOMETRIC_MARGIN

                    learned = True
                else:
                    # nothing captured; fall back to non-learned behaviour
                    learned = True
                    mean_gaze = np.array([0.0,0.0])
                    inv_cov = np.eye(2)

            # show progress (progress color changed)
            total_needed = len(CALIBRATION_STEPS)*FRAMES_PER_STEP
            cv2.putText(frame,f"Progress: {len(learning_samples)}/{total_needed}",
                        (30,260),cv2.FONT_HERSHEY_SIMPLEX,0.7,PROGRESS_COLOR,2)

            status = "CALIBRATING"
            confidence = 1.0 if len(learning_samples)>0 else 0.0

        else:
            # --- Operational phase ---
            if feats is None:
                status="NO_FACE"
                confidence=0.0
            else:
                dx_raw,dy_raw,yaw,pitch = feats

                dx=SMOOTHING*prev_dx+(1-SMOOTHING)*dx_raw
                dy=SMOOTHING*prev_dy+(1-SMOOTHING)*dy_raw
                prev_dx,prev_dy = dx,dy

                # Mahalanobis
                diff=np.array([dx,dy])-mean_gaze
                maha=np.sqrt(diff.T@inv_cov@diff)
                maha_score=max(0,1-maha/MAHALANOBIS_THRESHOLD)

                # Geometric gating (symmetric) with head pose
                horizontal_score=abs(dx)+HEAD_WEIGHT*abs(yaw)
                vertical_score=abs(dy)+HEAD_WEIGHT*abs(pitch)

                geometric_inside=(horizontal_score<=H_THRESHOLD and
                                  vertical_score<=V_THRESHOLD)

                if geometric_inside:
                    final_score=maha_score
                else:
                    final_score=0

                confidence=max(0,min(1,final_score))

                if final_score>0.3:
                    status="INSIDE"
                    outside_counter=0
                    mean_gaze=(1-ADAPT_RATE)*mean_gaze+ADAPT_RATE*np.array([dx,dy])
                else:
                    outside_counter+=1
                    if outside_counter>=OUTSIDE_CONFIRM_FRAMES:
                        status="OUTSIDE"
                    else:
                        status="INSIDE"

        # Logging
        writer.writerow([time.time(),status,round(float(confidence),3)])
        log.flush()

        # Visual overlays
        # only draw status/confidence overlay after calibration completed
        if learned:
            color = INSIDE_COLOR if status == "INSIDE" else NO_FACE_COLOR
            cv2.putText(frame, f"{status} | Conf:{confidence:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # show learned thresholds if available
        if learned:
            cv2.putText(frame,f"H_th:{H_THRESHOLD:.3f} V_th:{V_THRESHOLD:.3f}",
                        (30,320),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        cv2.imshow("ProctorGuard Hybrid 2.0",frame)

        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        # allow keyboard start as fallback
        if not learned and key==ord('s'):
            button_clicked = True

    cap.release()
    log.close()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run()
