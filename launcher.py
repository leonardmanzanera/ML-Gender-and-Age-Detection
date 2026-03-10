#!/usr/bin/env python3
"""
Antigravity Vision - Unified Launcher v2.0
"""

import sys
import subprocess
import os

def main():
    while True:
        print("\n==================================================")
        print("  🚀 ANTIGRAVITY VISION - PROFESSIONAL LAUNCHER 🚀")
        print("==================================================")
        print("Sélectionnez la version à lancer :")
        print(" [1] V1: Baseline (SSD + Caffe CNN)")
        print(" [2] V2: YOLOv8 Face + Caffe CNN")
        print(" [3] V3: YOLOv8 Face + Synchronous ViT/ONNX")
        print(" [4] V3.1: YOLOv8 Face + ASYNC ViT/ONNX (Optimal)")
        print(" [5] V4: General Object Detection (YOLOv8 COCO)")
        print(" [6] V5: UNIFIED VISION (Objects + Async ViT)")
        print(" [7] 🧠 ULTIMATE (Objects + ViT + Face ID + Posture + Privacy)")
        print(" [8] 🚀 V8 TRACKED (Multi-Person Optimised)")
        print(" [9] 🚨 V9 WATCHLIST (Most Wanted Face Matching)")
        print(" [10] ✨ V10 AESTHETICS (Golden Ratio & Symmetry)")
        print(" [q] Quitter")
        
        choice = input("\nVotre choix : ").strip().lower()
        
        pipeline_map = {
            '1': "v1_baseline.py",
            '2': "v2_yolo_caffe.py",
            '3': "v3_vit_onnx.py",
            '4': "v3_1_vit_async.py",
            '5': "v4_object_detection.py",
            '6': "v5_unified_vision.py",
            '7': "v_ultimate.py",
            '8': "v8_tracked.py",
            '9': "v9_watchlist.py",
            '10': "v10_beauty.py"
        }
        
        if choice in pipeline_map:
            script_path = os.path.join("pipelines", pipeline_map[choice])
            print(f"\n[*] Launching {pipeline_map[choice]}...")
            subprocess.run([sys.executable, script_path])
        elif choice == 'q':
            print("Goodbye.")
            break
        else:
            print("[!] Invalid choice.")

if __name__ == "__main__":
    main()
