import cv2
import numpy as np

def init_camera(indices=[0, 1, 2]):
    """
    Robust camera initialization for macOS.
    Detects and skips black frames (driver lock or permission issues).
    """
    cap = None
    for index in indices:
        print(f" -> Checking Camera (Index {index})...")
        temp_cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if temp_cap.isOpened():
            # Reading a test frame
            ret, frame = temp_cap.read()
            if ret and frame is not None:
                # Check for "black frame" (common macOS issue)
                if np.mean(frame) > 1.0:
                    cap = temp_cap
                    print(f" [+] Camera validated on index {index}.")
                    break
            temp_cap.release()
            
    if cap is None:
        print("[!] ERROR: No functional camera found.")
        print("    Check System Settings > Privacy & Security > Camera.")
        
    return cap
