class PerformanceLogger:
    """MLOps: Inference latency tracking."""
    def __init__(self):
        self.history = []
        
    def log(self, face_id, det_time, clf_time_1, clf_time_2=0, fps=0):
        total = det_time + clf_time_1 + clf_time_2
        metrics = {
            "face_id": face_id,
            "det_ms": round(det_time, 1),
            "clf1_ms": round(clf_time_1, 1),
            "clf2_ms": round(clf_time_2, 1),
            "total_ms": round(total, 1),
            "fps": round(fps, 1)
        }
        self.history.append(metrics)
