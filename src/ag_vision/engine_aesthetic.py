"""
AG Vision - Async Aesthetic Engine
Wraps AestheticEngine.analyze() in a background Thread + Queue.
Prevents blocking cv2.imshow() (~80ms inference).
"""

import threading
import collections
import copy
from ag_vision.aesthetic import AestheticEngine


class AsyncAestheticEngine:
    """
    Thread-safe wrapper around AestheticEngine.
    Inference runs in a background worker thread — never blocks the camera loop.

    API:
        submit(track_id, crop)  — queue a face crop for async analysis
        get_result(track_id)    — return latest cached result dict (or None)
        stop()                  — join the worker thread
    """

    MAX_QUEUE_SIZE = 4

    def __init__(self):
        self._engine = AestheticEngine()
        self.lock = threading.Lock()
        self.queue = collections.deque(maxlen=self.MAX_QUEUE_SIZE)
        self.results = {}          # {track_id: result_dict}
        self.is_running = True
        self.new_data_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        print("[*] AsyncAestheticEngine ready.")

    def submit(self, track_id, crop):
        """Queue a face crop for async aesthetic analysis."""
        with self.lock:
            self.queue.append((track_id, copy.deepcopy(crop)))
        self.new_data_event.set()

    def get_result(self, track_id):
        """Return the latest cached result for a track ID, or None."""
        with self.lock:
            return copy.deepcopy(self.results.get(track_id))

    def _worker_loop(self):
        while self.is_running:
            self.new_data_event.wait()
            if not self.is_running:
                break
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    track_id, crop = self.queue.popleft()
                result = self._engine.analyze(crop)
                if result is not None:
                    with self.lock:
                        self.results[track_id] = result
            self.new_data_event.clear()

    def stop(self):
        """Gracefully stop the worker thread."""
        self.is_running = False
        self.new_data_event.set()
        self.worker.join(timeout=2.0)
        print("[+] AsyncAestheticEngine stopped.")
