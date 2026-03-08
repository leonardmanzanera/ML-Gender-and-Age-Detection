import collections

class TemporalSmoother:
    """Statistical smoothing (Moving Mode/Mean) to stabilize age/gender predictions."""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.age_history = collections.defaultdict(lambda: collections.deque(maxlen=window_size))
        self.gender_history = collections.defaultdict(lambda: collections.deque(maxlen=window_size))
        
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
