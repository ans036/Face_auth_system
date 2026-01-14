class LivenessDetector:
    def __init__(self, min_frames=15):
        self.min_frames = min_frames

    def is_live(self, frames):
        return len(frames) >= self.min_frames
