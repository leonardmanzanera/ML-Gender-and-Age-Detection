import cv2
import time
import os
import sys
from ultralytics import YOLO

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ag_vision.config_utils import load_config
from ag_vision.camera import init_camera

def main():
    print("[*] Starting Pipeline V4: General Object Detection...")
    model = YOLO("yolov8n.pt")
    cap = init_camera()
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        results = model.predict(source=frame, conf=0.3, verbose=False)
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(frame, f"V4 Object Detection - FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf*100:.0f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("V4 Objects", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
