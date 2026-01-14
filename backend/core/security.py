import time

class ReplayGuard:
    def __init__(self, max_window=5):
        self.max_window = max_window

    def validate(self, timestamps):
        if not timestamps:
            return False
        return (max(timestamps) - min(timestamps)) <= self.max_window
