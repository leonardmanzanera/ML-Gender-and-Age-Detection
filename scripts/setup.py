import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ag_vision.utils.download_utils import download_file, get_model_path
from ultralytics import YOLO

# ── Model registry ──────────────────────────────────────────────────────────
# Each entry: filename → (download_url, subfolder)
# subfolder=None → models/
# subfolder="legacy" → models/legacy/

MODELS = {
    # Face detection (YOLOv8)
    "yolov8n-face.pt": (
        "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
        None
    ),
    # Age/Gender regression (Vision Transformer ONNX)
    "vit_age_gender.onnx": (
        "https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx",
        None
    ),
    # Gender classification (Caffe CNN — V3+)
    "gender_net.caffemodel": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel",
        None
    ),
    "gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
        None
    ),
    # Face detector SSD
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        None
    ),
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        None
    ),
    # Legacy Caffe age model (V1/V2 only — NOT used in V3+)
    "age_net.caffemodel": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel",
        "legacy"
    ),
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
        "legacy"
    ),
}


def main():
    print("")
    print("══════════════════════════════════════════════")
    print("  📦  AG Vision — Téléchargement des Modèles")
    print("══════════════════════════════════════════════")
    print("")

    # Ensure model directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/legacy", exist_ok=True)

    for filename, (url, subfolder) in MODELS.items():
        if subfolder:
            dest = get_model_path(os.path.join(subfolder, filename))
        else:
            dest = get_model_path(filename)
        download_file(url, dest)

    # YOLOv8 COCO (auto-downloaded by ultralytics on first use)
    print("")
    print("[*] Pré-chargement du modèle YOLOv8n (COCO 80 classes)...")
    YOLO(get_model_path("yolov8n.pt"))

    print("")
    print("[✓] Tous les modèles sont prêts.")
    print("")


if __name__ == "__main__":
    main()
