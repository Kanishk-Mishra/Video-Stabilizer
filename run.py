#!/usr/bin/env python3
import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

from pose import PoseEstimator, export_pose
from stabilization import compute_shifts
from rendering import ensure_dir, render_video

USE_YOLO=False
try:
    from ultralytics import YOLO
    USE_YOLO=True
except Exception:
    USE_YOLO=False

MOBILENET_PERSON_CLASS_ID = 15

def load_yolo_model():
    return YOLO("yolov8n.pt")

def load_mobilenet_ssd(net_dir=None):
    proto = os.path.join(net_dir, "MobileNetSSD_deploy.prototxt")
    model = os.path.join(net_dir, "MobileNetSSD_deploy.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto, model)
    return net

def detect_person_yolo(model, frame):
    res = model(frame, imgsz=(640,640), conf=0.3, device="cpu", verbose=False)[0]
    bboxes = []
    if getattr(res, "boxes", None) is None: return bboxes
    for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
        if int(cls)==0:
            bboxes.append(tuple(box.astype(int).tolist()))
    return bboxes

def detect_person_mobilenet(net, frame):
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,0.007843,(300,300),127.5)
    net.setInput(blob)
    det = net.forward()
    bboxes=[]
    for i in range(det.shape[2]):
        conf=float(det[0,0,i,2])
        cls=int(det[0,0,i,1])
        if conf>0.4 and cls==MOBILENET_PERSON_CLASS_ID:
            x1=int(det[0,0,i,3]*w); y1=int(det[0,0,i,4]*h)
            x2=int(det[0,0,i,5]*w); y2=int(det[0,0,i,6]*h)
            bboxes.append((x1,y1,x2,y2))
    return bboxes

def run_pipeline(input_video, out_dir, use_yolo=True, mobilenet_dir=None,
                 stabilize_x_only=False, export_side_by_side=True):
    t0=time.time()
    ensure_dir(out_dir)
    cap=cv2.VideoCapture(input_video)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=cap.get(cv2.CAP_PROP_FPS) or 30.0
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    yolo_model = load_yolo_model() if use_yolo and USE_YOLO else None
    mobilenet_net = None if yolo_model else load_mobilenet_ssd(mobilenet_dir)
    pose_est = PoseEstimator() if PoseEstimator else None

    frames=[]; centers=[]
    pose_keypoints=[]
    print("Scanning frames...")
    while True:
        ok, frame=cap.read()
        if not ok: break
        frames.append(frame.copy())
        # detection
        bboxes = detect_person_yolo(yolo_model, frame) if yolo_model else detect_person_mobilenet(mobilenet_net, frame)
        if bboxes:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
            x1,y1,x2,y2=bboxes[int(np.argmax(areas))]
            cx,cy=(x1+x2)//2,(y1+y2)//2
        else:
            cx,cy=np.nan,np.nan
        # pose
        if pose_est:
            pts=pose_est.infer(frame)
            pose_keypoints.append(pts)
            mid=PoseEstimator.mid_hip_from_pose(pts)
            if mid: centers.append([mid[0], mid[1]])
            else: centers.append([cx,cy])
        else:
            pose_keypoints.append({})
            centers.append([cx,cy])
    cap.release()

    centers=np.array(centers)
    for axis in (0,1):
        col=centers[:,axis]
        nan_mask=np.isnan(col)
        if nan_mask.all():
            centers[:,axis]=W//2 if axis==0 else H//2
        elif nan_mask.any():
            x=np.arange(len(col))
            good=~nan_mask
            centers[:,axis]=np.interp(x,x[good],col[good])

    shifts=compute_shifts(centers,W,H,stabilize_x_only)
    stabilized_path=os.path.join(out_dir, Path(input_video).stem+"_stabilized.mp4")
    sbs_path=os.path.join(out_dir, Path(input_video).stem+"_side_by_side.mp4") if export_side_by_side else None

    render_video(frames,shifts,stabilized_path,sbs_path,FPS)
    pose_csv, pose_json=export_pose(pose_keypoints,out_dir,Path(input_video).stem)
    t1=time.time()
    print("Done. FPS approx:", round(len(frames)/(t1-t0),2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True)
    p.add_argument("-o","--out_dir",default="outputs")
    p.add_argument("--no-yolo",action="store_true")
    p.add_argument("--mobilenet-dir",default=None)
    p.add_argument("--stabilize_x_only",action="store_true")
    p.add_argument("--no-side-by-side",action="store_true")
    args=p.parse_args()
    use_yolo = not args.no_yolo and USE_YOLO
    run_pipeline(args.input,args.out_dir,use_yolo,args.mobilenet_dir,
                 args.stabilize_x_only,not args.no_side_by_side)
