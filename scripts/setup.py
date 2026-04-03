import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ag_vision.utils.download_utils import download_file, get_model_path
from ultralytics import YOLO

def main():
    print("--- ANTIGRAVITY PROFESSIONAL SETUP ---")
    
    # 1. Weights
    models = {
        "yolov8n-face.pt": "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
        "vit_age_gender.onnx": "https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx",
        "age_net.caffemodel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel",
        "gender_net.caffemodel": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    os.makedirs("models", exist_ok=True)
    
    for filename, url in models.items():
        download_file(url, get_model_path(filename))
        
    # YOLO COCO
    print("[*] Pre-caching YOLOv8 COCO...")
    YOLO(get_model_path("yolov8n.pt"))
    
    print("\n[SUCCESS] Environment is fully provisioned.")

if __name__ == "__main__":
    main()
