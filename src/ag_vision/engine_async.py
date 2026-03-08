import cv2
import time
import os
import threading
import numpy as np
import copy
import onnxruntime as ort

class AsyncViTEngine:
    """
    Hybrid Asynchronous Engine:
    - ViT ONNX for age regression (excellent accuracy)
    - Caffe CNN for gender classification (proven reliable in V1/V2)
    """
    GENDER_LIST = ['Male', 'Female']
    
    def __init__(self, vit_model_path):
        print(f"[*] Threading: Initializing Hybrid Engine...")
        
        # ViT for age
        print(f"    -> ViT ONNX (Age): {os.path.basename(vit_model_path)}")
        self.vit_sess = ort.InferenceSession(vit_model_path, providers=['CPUExecutionProvider'])
        self.vit_input = self.vit_sess.get_inputs()[0].name
        
        # Caffe for gender
        models_dir = os.path.dirname(vit_model_path)
        gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(models_dir, "gender_net.caffemodel")
        
        if os.path.exists(gender_proto) and os.path.exists(gender_model):
            print(f"    -> Caffe CNN (Gender): gender_net.caffemodel")
            self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
            self.has_caffe_gender = True
        else:
            print(f"    [!] Caffe gender model not found. Using ViT fallback.")
            self.gender_net = None
            self.has_caffe_gender = False
        
        self.lock = threading.Lock()
        self.current_crop = None
        self.latest_result = {"age": "--", "gender": "Scanning", "gender_prob": 1.0, "time_ms": 0.0}
        
        self.is_running = True
        self.new_data_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def preprocess_vit(self, face_crop):
        """Preprocess for ViT (224x224, ImageNet normalization)."""
        try:
            img = cv2.resize(face_crop, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img
        except Exception:
            return None
    
    def classify_gender_caffe(self, face_crop):
        """Classify gender using the proven Caffe CNN model."""
        try:
            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            self.gender_net.setInput(blob)
            preds = self.gender_net.forward()
            gender_idx = preds[0].argmax()
            confidence = float(preds[0][gender_idx])
            return self.GENDER_LIST[gender_idx], confidence
        except Exception:
            return "Unknown", 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def submit(self, crop):
        if not self.new_data_event.is_set():
            with self.lock:
                self.current_crop = copy.deepcopy(crop)
            self.new_data_event.set()

    def get_latest(self):
        with self.lock:
            return copy.deepcopy(self.latest_result)

    def _worker_loop(self):
        while self.is_running:
            self.new_data_event.wait()
            if not self.is_running: break
            
            with self.lock:
                crop = self.current_crop
            
            if crop is not None:
                t0 = time.time()
                
                # --- AGE: ViT ONNX (always) ---
                tensor = self.preprocess_vit(crop)
                raw_age = "--"
                if tensor is not None:
                    preds = self.vit_sess.run(None, {self.vit_input: tensor})[0][0]
                    raw_age = preds[0]
                
                # --- GENDER: Caffe CNN (reliable) or ViT fallback ---
                if self.has_caffe_gender:
                    gen_str, gen_conf = self.classify_gender_caffe(crop)
                else:
                    # ViT fallback (biased, but better than nothing)
                    if tensor is not None:
                        gender_prob = self.sigmoid(preds[1])
                        if gender_prob > 0.5:
                            gen_str, gen_conf = "Female", float(gender_prob)
                        else:
                            gen_str, gen_conf = "Male", float(1.0 - gender_prob)
                    else:
                        gen_str, gen_conf = "Unknown", 0.0
                
                t_clf = (time.time() - t0) * 1000
                    
                with self.lock:
                    self.latest_result = {
                        "age": raw_age, "gender": gen_str, 
                        "gender_prob": gen_conf, "time_ms": t_clf
                    }
            self.new_data_event.clear()

    def stop(self):
        self.is_running = False
        self.new_data_event.set()
        self.worker.join(timeout=1.0)
