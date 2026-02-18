ProctorGuard Hybrid — Webcam Gaze Detection
==========================================

Brief
-----
ProctorGuard Hybrid is a webcam‑based gaze detection / proctoring helper that fuses a learned gaze model (OpenVINO gaze-estimation) with geometric gating and head-pose information. The current implementation uses an interactive, per-step calibration: the user presses a Start button for each calibration target (CENTER, LEFT, RIGHT, TOP, BOTTOM). The program logs status (INSIDE/OUTSIDE), a confidence score, and timestamps to a CSV.

This README explains required models and environment setup, how to run the script, calibration flow and controls, CSV output format, and common troubleshooting tips.

Quick features
--------------
- Interactive per-step calibration (user presses Start for each target).
- Head pose captured during calibration and used to improve thresholds.
- Fusion of Mahalanobis distance and geometric gating for INSIDE/OUTSIDE decision.
- Live webcam processing (frame resizing controlled in the script) with minimal UI.
- Output CSV log: `gaze_log.csv` (timestamp, status, confidence).

Repository files (important)
---------------------------
- `chunks.py` — Main script you run. Implements calibration, model loading, inference, overlays and logging.
- `intel/` — Directory where OpenVINO IR models should be placed. See "Required models" below.
- `requirement.txt` — (if present) Python dependencies; otherwise see the Requirements section.
- `gaze_log.csv` — Output log written when the program runs.

Requirements
------------
- Windows 10/11 (tested in this workspace). Mac/Linux should also work but file paths and camera backend differ.
- Python 3.8+ 
- OpenVINO runtime (pip package `openvino` matching your environment).
- OpenCV (cv2)
- NumPy

Recommended pip install (inside a virtualenv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirement.txt
# if requirement.txt is missing, install at least:
pip install openvino openvino-dev opencv-python numpy
```

Required models (place under `intel/`)
-------------------------------------
Download OpenVINO IR models (FP32) from the OpenVINO Model Zoo and place them under: `intel/<model-name>/FP32/`.
The script expects these specific IR file paths (XML + BIN):

- intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
- intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml
- intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml
- intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml

If you get FileNotFoundError referencing an xml path, download the correct IR (XML+BIN) and place it at the above path.

Run instructions
----------------
From the project root (where `chunks.py` is located):

```powershell
# activate venv if you created one
.\.venv\Scripts\Activate.ps1
python chunks.py
```

Controls and calibration flow
-----------------------------
- On start the script opens a small UI window named "ProctorGuard Hybrid 2.0".
- The program will prompt: "Calibration: Look at <TARGET>" where TARGET cycles through CENTER, LEFT, RIGHT, TOP, BOTTOM.
- For each target the window shows a green START button. Press the button with the mouse or press the `s` key as a fallback.
- After you press START the program waits WAIT_BEFORE_CAPTURE (default 2.0 s) shown as a countdown, then captures FRAMES_PER_STEP frames for that target.
- Repeat pressing START for each target until the calibration finishes.
- After calibration the program switches to operational mode and overlays the status (INSIDE / OUTSIDE) and confidence only after calibration is complete.
- Press `q` to quit the program.

Tuning (in `chunks.py`)
-----------------------
Open `chunks.py` and change these constants near the top:
- FRAME_WIDTH — controls the processing width (affects speed and coordinate scaling). Set to your camera capture width if you want full-resolution processing; also change camera capture properties if needed.
- FRAMES_PER_STEP — how many frames to capture for each calibration target.
- WAIT_BEFORE_CAPTURE — how long the script waits after pressing Start before capturing frames.
- MAHALANOBIS_THRESHOLD, HEAD_WEIGHT, GEOMETRIC_MARGIN — tuning constants for the fusion and thresholds.

If edges appear misaligned or INSIDE/OUTSIDE classifications look shifted, try increasing FRAMES_PER_STEP and ensure you look precisely at the requested calibration points (center, left edge of monitor, etc.).

CSV output format and meaning
-----------------------------
The script writes to `gaze_log.csv` in the project root. Current columns are:
- timestamp — epoch time in seconds (float) when the line was written
- status — string: CALIBRATING / NO_FACE / INSIDE / OUTSIDE
- confidence — float in [0,1]

Note: during calibration the script also logs lines with status `CALIBRATING` while collecting samples. After calibration the lines show INSIDE/OUTSIDE (or NO_FACE).

Troubleshooting
---------------
- "Model XML not found" errors: make sure the requested model XML/BIN are placed in the expected `intel/<name>/FP32/` folder.
- OpenVINO runtime import errors: ensure you installed `openvino` in the same Python interpreter used to run the script.
- Camera resolution / backend issues on Windows: if you need full sensor resolution, set the capture property explicitly in the script before the read loop, for example:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

Place those calls immediately after `cv2.VideoCapture(0)` in `chunks.py`.

- If the program shows "NO_FACE" frequently, improve room lighting and ensure the face is fully inside the webcam frame. If face detection finds a small region, the landmarks/gaze model may produce noisy outputs.

- If every gaze is detected OUTSIDE or INSIDE regardless of orientation:
  - Re-run the calibration and follow prompts precisely.
  - Increase FRAMES_PER_STEP to average more samples.
  - Inspect the numeric values logged (you can open `gaze_log.csv` during/after a run) and compare them to the thresholds.

Notes about coordinate systems and accuracy
------------------------------------------
- The OpenVINO gaze-estimation model returns a 2D gaze vector (x,y) in a camera-relative coordinate system. The script uses per-user calibration (mean and covariance) plus simple geometric gating that uses absolute gaze components and head-pose to compute thresholds.
- If you need screen-mapping (to map gaze to pixel coordinates), that requires an explicit mapping step (e.g., homography or regression) between calibration points and your physical screen coordinates. The present script only classifies INSIDE vs OUTSIDE relative to learned gaze distribution.

Extending or modifying the script
--------------------------------
- To process at full camera resolution but keep the UI small: set the capture properties with `cap.set(...)` to the desired width/height; in `get_features()` you can still scale the frame for processing by changing FRAME_WIDTH or removing the resize.
- To save richer output (gaze_x,gaze_y,yaw,pitch,radial distance, etc.), modify the logging writer in `chunks.py` to include those fields.

