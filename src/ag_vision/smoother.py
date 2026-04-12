import collections

class TemporalSmoother:
    """
    Statistical smoothing (Moving Mode/Mean) to stabilize age/gender predictions.
    
    Supports two modes per track_id:
        - Continuous (ViT regression): Moving Average over window
        - Categorical (Caffe bins): Moving Mode + averaged probability
    
    Used by TrackedViTEngine for per-ID temporal stabilization.
    """
    def __init__(self, window_size=8):
        self.window_size = window_size
        self.age_history = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )
        self.gender_history = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )
        
    def _get_mode(self, d):
        if not d:
            return None, 0.0
        
        # Find most frequent category
        counts = collections.Counter([item[0] for item in d])
        mode_val = counts.most_common(1)[0][0]
        
        # Calculate avg probability for this mode
        probs = [item[1] for item in d if item[0] == mode_val]
        avg_prob = sum(probs) / len(probs) if probs else 0.0
        
        return mode_val, avg_prob
        
    def update_and_get(self, track_id, pre_age, prob_age, pre_gen, prob_gen, is_regression=False):
        """
        Update smoothing buffers and return stabilized predictions.
        
        Args:
            track_id: Unique ID for this tracked face
            pre_age: Raw age prediction (float for ViT, str for Caffe bins)
            prob_age: Age confidence (unused in regression mode)
            pre_gen: Raw gender string ("Male" / "Female")
            prob_gen: Gender confidence
            is_regression: True for ViT continuous age, False for Caffe bins
            
        Returns:
            (smoothed_age_str, age_prob, smoothed_gender, gender_prob)
        """
        if is_regression:
            # For continuous age (ViT)
            self.age_history[track_id].append(pre_age)
            s_age = int(sum(self.age_history[track_id]) / len(self.age_history[track_id]))
            s_prob_age = prob_age
        else:
            # For binned age (Caffe)
            self.age_history[track_id].append((pre_age, prob_age))
            s_age, s_prob_age = self._get_mode(self.age_history[track_id])

        self.gender_history[track_id].append((pre_gen, prob_gen))
        s_gen, s_prob_gen = self._get_mode(self.gender_history[track_id])
        
        return str(s_age), s_prob_age, s_gen, s_prob_gen

    def purge(self, track_id):
        """Remove all history for a given track_id (called on stale ID cleanup)."""
        self.age_history.pop(track_id, None)
        self.gender_history.pop(track_id, None)
