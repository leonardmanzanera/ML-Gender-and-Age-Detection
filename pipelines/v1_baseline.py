import cv2
import time
import os
import sys

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ag_vision.config_utils import load_config
from ag_vision.logger import PerformanceLogger
from ag_vision.smoother import TemporalSmoother
from ag_vision.camera import init_camera
from ag_vision.utils.download_utils import get_model_path

def compile_caffe_network(prototxt, caffemodel):
    p_path, c_path = get_model_path(prototxt), get_model_path(caffemodel)
    net = cv2.dnn.readNetFromCaffe(p_path, c_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def main():
    print("[*] Starting Pipeline V1: Baseline (SSD + Caffe)...")
    config = load_config()
    ui = config["ui"]
    
    net_ssd = compile_caffe_network(config["model"]["ssd"]["prototxt"], config["model"]["ssd"]["caffemodel"])
    net_age = compile_caffe_network(config["model"]["age"]["prototxt"], config["model"]["age"]["caffemodel"])
    net_gender = compile_caffe_network(config["model"]["gender"]["prototxt"], config["model"]["gender"]["caffemodel"])
    
    logger = PerformanceLogger()
    smoother = TemporalSmoother(window_size=config["smoothing"]["window_size"])
    cap = init_camera()
    
    if not cap: return

    AGE_BINS = config["model"]["age"]["bins"]
    GENDER_CATS = config["model"]["gender"]["categories"]
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Detection
        ssd_cfg = config["model"]["ssd"]
        blob = cv2.dnn.blobFromImage(frame, ssd_cfg["scale_factor"], tuple(ssd_cfg["input_size"]), tuple(ssd_cfg["mean_values"]))
        net_ssd.setInput(blob)
        t0 = time.time()
        detections = net_ssd.forward()
        t_ssd = (time.time() - t0) * 1000
        
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(display_frame, f"V1 Baseline - FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > ssd_cfg["confidence_threshold"]:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face_crop.size == 0: continue
                
                # Classifiers
                face_blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.4, 87.7, 114.8), swapRB=False)
                
                net_gender.setInput(face_blob)
                t1 = time.time()
                g_preds = net_gender.forward()
                t_gen = (time.time() - t1) * 1000
                
                net_age.setInput(face_blob)
                t2 = time.time()
                a_preds = net_age.forward()
                t_age = (time.time() - t2) * 1000
                
                s_age, _, s_gen, s_gen_prob = smoother.update_and_get(
                    "Face0", AGE_BINS[a_preds.argmax()], 1.0, GENDER_CATS[g_preds.argmax()], g_preds.max()
                )
                
                logger.log(0, t_ssd, t_age, t_gen, fps)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, f"{s_gen}, {s_age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Antigravity Professional V1", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
