"""
AG Vision - Face Registry (V10: Reconnaissance Nommée)
Manages a persistent database of known face embeddings for real-time identification.
Uses dlib/face_recognition for 128-dimensional face encodings.
"""

import os
import pickle
import threading
import copy
import numpy as np

try:
    import face_recognition
except ImportError:
    print("[!] face_recognition not installed. Run: pip install face_recognition")
    face_recognition = None


class FaceRegistry:
    """
    Persistent face embedding database.
    Stores 128-dim encodings per person in a pickle file.
    Supports few-shot registration (1-3 photos per person).
    """
    
    TOLERANCE = 0.60  # Standard dlib tolerance
    
    def __init__(self, db_path=None):
        if db_path is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_path = os.path.join(root, "data", "known_faces.pkl")
        
        self.db_path = db_path
        self.lock = threading.Lock()
        self.known_names = []
        self.known_encodings = []
        self._load()
    
    def _load(self):
        """Load existing face database from disk."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                self.known_names = data.get("names", [])
                self.known_encodings = data.get("encodings", [])
            print(f" [+] FaceRegistry: Loaded {len(self.known_names)} known face(s).")
        else:
            print(" [i] FaceRegistry: No database found. Starting fresh.")
    
    def _save(self):
        """Persist the face database to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "names": self.known_names,
                "encodings": self.known_encodings
            }, f)
        print(f" [+] FaceRegistry: Saved {len(self.known_names)} face(s) to disk.")
    
    def register(self, name, bgr_frame):
        """
        Register a new face from a BGR frame (OpenCV format).
        Returns True if a face was found and registered, False otherwise.
        """
        if face_recognition is None:
            return False
            
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb = bgr_frame[:, :, ::-1]
        
        # Ensure frame is contiguous and uint8
        rgb = np.ascontiguousarray(rgb).astype(np.uint8)
        
        # Detect face locations
        locations = face_recognition.face_locations(rgb, model="hog")
        if len(locations) == 0:
            print(f" [!] No face detected for '{name}'. Try again.")
            return False
        
        # Compute encodings (using num_jitters=1 explicitly or 0 if it fails)
        try:
            encodings = face_recognition.face_encodings(rgb, locations, num_jitters=1)
        except TypeError:
            # Fallback if pybind11 is strictly matching signature 1 vs 2
            print(" [i] FaceRegistry: Signature mismatch, retrying with num_jitters=0...")
            encodings = face_recognition.face_encodings(rgb, locations, num_jitters=0)
        if len(encodings) == 0:
            return False
        
        with self.lock:
            self.known_names.append(name)
            self.known_encodings.append(encodings[0])
            self._save()
        
        print(f" [+] Registered '{name}' successfully!")
        return True
    
    def identify(self, bgr_crop):
        """
        Identify a face crop against the known database.
        Returns (name, distance) if matched, ("Unknown", 1.0) otherwise.
        """
        if face_recognition is None or len(self.known_encodings) == 0:
            return "Unknown", 1.0
        
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb = bgr_crop[:, :, ::-1]
        
        # Ensure frame is contiguous and uint8
        rgb = np.ascontiguousarray(rgb).astype(np.uint8)

        try:
            # Use num_jitters=0 for real-time identification
            encodings = face_recognition.face_encodings(rgb, num_jitters=0)
            if len(encodings) == 0:
                return "Unknown", 1.0
            
            encoding = encodings[0]
            
            with self.lock:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
            
            if len(distances) == 0:
                return "Unknown", 1.0
            
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            
            if best_distance < self.TOLERANCE:
                return self.known_names[best_idx], float(best_distance)
            else:
                return "Unknown", float(best_distance)
                
        except Exception as e:
            print(f" [!] FaceRegistry Error in identify: {e}")
            return "Unknown", 1.0
    
    def clear(self):
        """Reset the face database."""
        with self.lock:
            self.known_names = []
            self.known_encodings = []
            self._save()
        print(" [!] FaceRegistry: Database cleared.")

    def list_known(self):
        """Returns a list of registered names."""
        with self.lock:
            return list(set(self.known_names))
    
    def count(self):
        """Returns the number of registered face encodings."""
        return len(self.known_encodings)
