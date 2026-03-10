"""
AG Vision - Watchlist Engine (V9)
Manages an offline database of "Most Wanted" faces.
Calculates similarity percentage for given crops.
"""

import os
import cv2
import numpy as np
import threading

try:
    import face_recognition
except ImportError:
    print("[!] face_recognition not installed. Watchlist disabled.")
    face_recognition = None

class Watchlist:
    def __init__(self, watch_dir=None):
        if watch_dir is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            watch_dir = os.path.join(root, "data", "watchlist")
        
        self.watch_dir = watch_dir
        self.targets = {}  # {name: [encoding1, encoding2, ...]}
        self.lock = threading.Lock()
        
        if face_recognition is not None:
            self._load_watchlist()
            
    def _load_watchlist(self):
        if not os.path.exists(self.watch_dir):
            os.makedirs(self.watch_dir, exist_ok=True)
            print(f" [i] Watchlist: Created empty directory at {self.watch_dir}")
            return
            
        print(" [*] Watchlist: Scanning for targets...")
        for target_name in os.listdir(self.watch_dir):
            if target_name.startswith('.'):
                continue
                
            target_path = os.path.join(self.watch_dir, target_name)
            if not os.path.isdir(target_path):
                continue
                
            encodings = []
            for img_name in os.listdir(target_path):
                if img_name.startswith('.'):
                    continue
                    
                img_path = os.path.join(target_path, img_name)
                try:
                    # face_recognition loads image in RGB format
                    image = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    
                    if len(face_locations) > 0:
                        face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=1)
                        if len(face_encodings) > 0:
                            encodings.append(face_encodings[0])
                except Exception as e:
                    print(f" [!] Watchlist: Error loading {img_name} for target {target_name}: {e}")
                    
            if encodings:
                self.targets[target_name] = encodings
                print(f" [+] Watchlist: Loaded '{target_name}' with {len(encodings)} reference images.")
            else:
                print(f" [!] Watchlist: No valid faces found for '{target_name}' in {target_path}")

    def compare(self, bgr_crop):
        """
        Compare a BGR crop to the watchlist.
        Returns (best_match_name, score_percentage).
        If no match, returns (None, 0.0)
        """
        if face_recognition is None or len(self.targets) == 0:
            return None, 0.0
            
        rgb = bgr_crop[:, :, ::-1]
        rgb = np.ascontiguousarray(rgb).astype(np.uint8)
        
        try:
            # Use num_jitters=0 for speed in real-time
            encodings = face_recognition.face_encodings(rgb, num_jitters=0)
            if len(encodings) == 0:
                return None, 0.0
                
            encoding = encodings[0]
            
            best_name = None
            best_score = 0.0
            
            with self.lock:
                for target_name, target_encodings in self.targets.items():
                    distances = face_recognition.face_distance(target_encodings, encoding)
                    if len(distances) == 0:
                        continue
                        
                    min_distance = float(np.min(distances))
                    
                    # Convert distance to a percentage score
                    # A distance of 0.0 -> 100%, 0.4 -> 60%, 0.6 -> 40%, 1.0 -> 0%
                    score = max(0.0, (1.0 - min_distance) * 100.0)
                        
                    if score > best_score:
                        best_score = score
                        best_name = target_name
                        
            return best_name, round(best_score, 1)
            
        except Exception:
            return None, 0.0
