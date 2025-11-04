import cv2
import numpy as np
import pandas as pd
import json

try:
    import mediapipe as mp
except Exception:
    mp = None

class PoseEstimator:
    def __init__(self):
        if mp is None:
            raise ImportError("mediapipe not installed. `pip install mediapipe`")
        self.pose = mp.solutions.pose.Pose(model_complexity=1, enable_segmentation=False)

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        pts = {}
        if res is None or res.pose_landmarks is None:
            return pts
        lm = res.pose_landmarks.landmark
        for i, l in enumerate(lm):
            x = int(np.clip(l.x * w, 0, w-1))
            y = int(np.clip(l.y * h, 0, h-1))
            v = float(getattr(l, "visibility", 0.0))
            pts[f"kp_{i}"] = (x, y, v)
        return pts

    @staticmethod
    def mid_hip_from_pose(pts):
        L = pts.get("kp_23")
        R = pts.get("kp_24")
        def ok(p): return p is not None and p[2] > 0.4
        if ok(L) and ok(R):
            cx = (L[0] + R[0]) // 2
            cy = (L[1] + R[1]) // 2
            return (int(cx), int(cy))
        vis = [(x,y) for (x,y,v) in pts.values() if v > 0.4]
        if vis:
            arr = np.array(vis)
            return (int(arr[:,0].mean()), int(arr[:,1].mean()))
        return None

def export_pose(pose_keypoints, out_dir, video_stem, pose_csv_name=None, pose_json_name=None):
    pose_csv = f"{out_dir}/{pose_csv_name or video_stem + '_pose.csv'}"
    pose_json = f"{out_dir}/{pose_json_name or video_stem + '_pose.json'}"
    rows = []
    all_keys = set()
    for i, pk in enumerate(pose_keypoints):
        row = {"frame": i}
        for k, v in pk.items():
            row[f"{k}_x"] = v[0]; row[f"{k}_y"] = v[1]; row[f"{k}_v"] = v[2]
            all_keys.update([f"{k}_x", f"{k}_y", f"{k}_v"])
        rows.append(row)
    keys = ["frame"] + sorted(all_keys)
    df_rows = [{k: r.get(k,"") for k in keys} for r in rows]
    pd.DataFrame(df_rows)[keys].to_csv(pose_csv, index=False)
    with open(pose_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return pose_csv, pose_json
