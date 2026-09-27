import cv2
import torch

# ========== RENDER FIX FOR PYTORCH 2.6 ==========
try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import Conv, C2f, SPPF, Bottleneck, DFL, Detect, Concat, C1, C3
    import ultralytics.nn.tasks
    import ultralytics.nn.modules.block
    torch.serialization.add_safe_globals([
        DetectionModel, Conv, C2f, SPPF, Bottleneck, DFL, Detect, Concat, C1, C3,
        ultralytics.nn.tasks.DetectionModel
    ])
except Exception as e:
    print(f"Safe globals warning: {e}")

# Patch torch.load to allow YOLO checkpoint
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ========== END FIX ==========

from ultralytics import YOLO
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app, origins="*")

print("Loading YOLO... this takes 20 sec first time")
model = YOLO('yolov8n.pt')
COCO = model.names
print("YOLO Loaded! - RAKSHAK V6 READY")

latest_data = {
    "human_detected": False,
    "confidence": 0,
    "gas_ppm": 95,
    "obstacles": [],
    "turn": "STRAIGHT - Path Clear",
    "status": "Scanning...",
    "brightness": 0
}

def enhance_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bright = float(np.mean(gray))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    if mean_bright < 40:
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
        alpha, beta, gamma = 2.2, 60, 1.1
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        alpha, beta, gamma = 1.2, 15, 1.4
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced = np.power(enhanced/255.0, 1.0/gamma) * 255.0
    enhanced = enhanced.astype(np.uint8)
    return cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta), mean_bright

@app.route('/')
def home():
    return jsonify({"status":"RAKSHAK LIVE - Jharia Mine","endpoints":["/api/detect","/api/status"]})

@app.route('/api/status')
def status_api():
    return jsonify(latest_data)

@app.route('/api/detect', methods=['POST'])
def detect():
    global latest_data
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error":"No image"}), 400

        img_str = data['image'].split(',')[-1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error":"Invalid image"}), 400

        enhanced, mean_bright = enhance_dark(frame)
        conf_thres = 0.15 if mean_bright < 50 else 0.25
        results = model(enhanced, verbose=False, conf=conf_thres)

        fw, fh = frame.shape[1], frame.shape[0]
        best_person = None
        max_conf = 0
        obstacles = []
        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == 0 and conf > 0.15 and (x2-x1) > 35:
                    if best_person is None or (x2-x1)*(y2-y1) > best_person.get('area',0):
                        best_person = {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf,"area":(x2-x1)*(y2-y1)}
                        max_conf = conf
                elif conf > 0.35:
                    obstacles.append({"name":COCO[cls],"x":(x1+x2)//2})
                    detections.append({"type":"obstacle","x1":x1,"y1":y1,"x2":x2,"y2":y2,"label":COCO[cls],"conf":conf})

        if best_person:
            detections.append({"type":"person","x1":best_person["x1"],"y1":best_person["y1"],"x2":best_person["x2"],"y2":best_person["y2"],"conf":best_person["conf"],"label":"PERSON"})
            latest_data["human_detected"] = True
            latest_data["confidence"] = int(max_conf*100)
            latest_data["status"] = f"Human Signature Detected - {int(max_conf*100)}%"
        else:
            latest_data["human_detected"] = False
            latest_data["confidence"] = 0
            latest_data["status"] = "Scanning for survivors..."

        latest_data["obstacles"] = obstacles[:4]
        latest_data["brightness"] = int(mean_bright)
        latest_data["turn"] = "STRAIGHT - Path Clear"

        return jsonify({"detections":detections, "status":latest_data})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error":str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
