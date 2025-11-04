import os
import cv2
import numpy as np

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def render_video(frames, shifts, out_path, side_by_side_path=None, FPS=30):
    N = len(frames)
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (W,H))
    vw_sbs = cv2.VideoWriter(side_by_side_path, fourcc, FPS, (2*W,H)) if side_by_side_path else None
    for i,(frame,sh) in enumerate(zip(frames, shifts)):
        dx, dy = float(round(sh[0])), float(round(sh[1]))
        M = np.float32([[1,0,dx],[0,1,dy]])
        out = cv2.warpAffine(frame, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        vw.write(out)
        if vw_sbs is not None:
            try:
                sbs = np.hstack([frame, out])
            except Exception:
                outR = cv2.resize(out, (frame.shape[1], frame.shape[0]))
                sbs = np.hstack([frame, outR])
            vw_sbs.write(sbs)
    vw.release()
    if vw_sbs: vw_sbs.release()
