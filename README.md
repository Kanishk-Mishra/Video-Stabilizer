# Human-Centric Subject-Locked Video Stabilizer

## Overview
This project stabilizes videos of humans walking by locking the subject to a target point (center by default) while allowing the background to drift naturally. It also extracts pose keypoints and produces side-by-side comparisons.

**Features:**
- Person detection: YOLOv8 (ultralytics) or MobileNet-SSD fallback.
- Pose estimation: MediaPipe Pose (33 landmarks).
- Robust smoothing: Median + Savitzky-Golay filters.
- Outputs:
  - Stabilized video
  - Side-by-side comparison (original | stabilized)
  - Pose keypoints CSV & JSON
  - Metadata JSON (runtime, FPS, frame size)

---

## Setup
```bash
git clone <repo_url>
cd <repo>/src
python3 -m venv venv
# Activate virtual environment:
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r src/requirements.txt
```

### MobileNet-SSD fallback
Download these files and provide their path with `--mobilenet-dir`:

- [MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt?utm_source=chatgpt.com)  
- [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel?utm_source=chatgpt.com)

---

## Usage
```bash
python run.py -i input.mp4 -o outputs
```

### Options
- `--no-yolo` : disables YOLO and uses MobileNet-SSD  
- `--mobilenet-dir <dir>` : path to MobileNet model files  
- `--stabilize_x_only` : stabilize only horizontal movement  
- `--no-side-by-side` : disable side-by-side output  

---

## Methods
1. **Detection**: YOLOv8 or MobileNet-SSD detects bounding boxes of humans.  
2. **Pose**: MediaPipe Pose extracts landmarks; mid-hip used as subject center.  
3. **Center selection**: Prefer pose center; fallback to bbox center; interpolate missing frames.  
4. **Smoothing**: Median + Savitzky-Golay filter produce stable camera path.  
5. **Warping**: Translate frames to lock subject to target point.  
6. **Rendering**: Stabilized and side-by-side videos generated; pose data exported.  

---

## Runtime
Tested on CPU (Intel i7-9700K) on a 1080p, 300-frame video:

- Approx. 10-12 FPS with YOLOv8 CPU  
- Approx. 15-18 FPS with MobileNet-SSD  

GPU support improves FPS significantly.

---

## Limitations
- Only translation-based stabilization; rotation/zoom not handled.  
- Works best for single walking subject; multiple people may reduce accuracy.  
- Background removal not yet implemented.  
- YOLOv8 CPU is slow; GPU recommended for real-time use.
