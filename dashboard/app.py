import os
import sys
import time
import base64
import pathlib
import threading
from pathlib import Path
from io import BytesIO

# Windows path fix (kept from original project)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, render_template, request, jsonify, Response
import torch
import cv2
import numpy as np
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # PROJECT root
sys.path.insert(0, str(BASE_DIR))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors

# ── config ──────────────────────────────────────────────────────────────────
WEIGHTS = BASE_DIR / "best.pt"
DATA_YAML = BASE_DIR / "data" / "data.yaml"
IMG_SIZE  = 640
CONF      = 0.05
IOU       = 0.45
DEVICE    = ""

app = Flask(__name__)

# ── load model once ──────────────────────────────────────────────────────────
print("Loading model…")
device = select_device(DEVICE)
model  = DetectMultiBackend(str(WEIGHTS), device=device, data=str(DATA_YAML))
stride, names, pt = model.stride, model.names, model.pt
imgsz  = check_img_size(IMG_SIZE, s=stride)
model.warmup(imgsz=(1, 3, imgsz, imgsz))
print(f"Model loaded — classes: {names}")

# ── detection stats (in-memory) ──────────────────────────────────────────────
stats = {
    "total_images": 0,
    "total_detections": 0,
    "sessions": 0,
    "last_conf": 0.0,
    "history": []          # list of {time, count, source}
}
stats_lock = threading.Lock()

# ── helpers ──────────────────────────────────────────────────────────────────
def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    return img.unsqueeze(0)

def run_inference(img_bgr, conf_thres=CONF):
    orig_h, orig_w = img_bgr.shape[:2]
    tensor = preprocess(img_bgr)
    with torch.no_grad():
        pred = model(tensor)
    pred = non_max_suppression(pred, conf_thres, IOU)[0]

    detections = []
    annotator = Annotator(img_bgr.copy(), line_width=2)

    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(tensor.shape[2:], pred[:, :4],
                                   img_bgr.shape).round()
        for *xyxy, conf, cls in pred:
            c     = int(cls)
            label = f"{names[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))
            detections.append({
                "class": names[c],
                "confidence": round(float(conf), 3),
                "bbox": [int(x) for x in xyxy]
            })

    annotated = annotator.result()
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()
    return b64, detections

def record(count, source, conf):
    with stats_lock:
        stats["total_detections"] += count
        stats["total_images"]     += 1
        stats["last_conf"]         = conf
        stats["history"].append({
            "time":   time.strftime("%H:%M:%S"),
            "count":  count,
            "source": source
        })
        if len(stats["history"]) > 50:
            stats["history"] = stats["history"][-50:]

# ── webcam stream ────────────────────────────────────────────────────────────
webcam_active = False
webcam_lock   = threading.Lock()

def gen_frames(conf_thres):
    global webcam_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    with webcam_lock:
        webcam_active = True
    try:
        while webcam_active:
            ok, frame = cap.read()
            if not ok:
                break
            b64, dets = run_inference(frame, conf_thres)
            record(len(dets), "webcam", conf_thres)
            # decode back to jpeg for MJPEG stream
            raw = base64.b64decode(b64)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + raw + b"\r\n")
    finally:
        cap.release()
        with webcam_lock:
            webcam_active = False

# ── routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_classes=names)

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    conf = float(request.form.get("conf", CONF))
    file = request.files["image"]
    img  = Image.open(file.stream).convert("RGB")
    arr  = np.array(img)[:, :, ::-1].copy()   # RGB→BGR

    b64, dets = run_inference(arr, conf)
    record(len(dets), "upload", conf)

    return jsonify({
        "image":      b64,
        "detections": dets,
        "count":      len(dets)
    })

@app.route("/webcam_feed")
def webcam_feed():
    conf = float(request.args.get("conf", CONF))
    return Response(gen_frames(conf),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/webcam_stop", methods=["POST"])
def webcam_stop():
    global webcam_active
    with webcam_lock:
        webcam_active = False
    return jsonify({"stopped": True})

@app.route("/stats")
def get_stats():
    with stats_lock:
        return jsonify(dict(stats))

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
