# utils/page_hinkley.py
import numpy as np

class PageHinkley:
    """
    Page-Hinkley Test for online drift detection.
    Detects significant changes in a monitored metric (e.g., reconstruction error).
    """

    def __init__(self, delta=1.0, lambd=500, min_instances=100, verbose=False):
        """
        Args:
            delta (float): Small offset to reduce sensitivity to small fluctuations.
            lambd (float): Drift threshold (higher → fewer alarms).
            min_instances (int): Warm-up before drift checking.
            verbose (bool): Print debug logs.
        """
        self.delta = float(delta)
        self.lambd = float(lambd)
        self.min_instances = int(min_instances)
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Reset internal statistics."""
        self.sum = 0.0
        self.x_mean = 0.0
        self.num_instances = 0
        self.min_sum = 0.0
        self.drift_detected = False

    def update(self, error):
        """
        Feed one observation and update drift statistics.
        Returns True if drift detected.
        """
        try:
            x = float(error)
        except Exception:
            # silently ignore pathological values
            return False

        # increment counter
        self.num_instances += 1

        # --- always update mean ---
        old_mean = self.x_mean
        self.x_mean += (x - self.x_mean) / self.num_instances

        # --- update cumulative deviation ---
        self.sum += x - (old_mean + self.delta)

        # track minimum
        if self.sum < self.min_sum:
            self.min_sum = self.sum

        if self.verbose:
            print(f"[PageHinkley] n={self.num_instances}, x={x:.6f}, mean={self.x_mean:.6f}, "
                  f"sum={self.sum:.6f}, min_sum={self.min_sum:.6f}, lambd={self.lambd}")

        # warm-up period
        if self.num_instances < self.min_instances:
            return False

        # drift condition
        self.drift_detected = (self.sum - self.min_sum) > self.lambd
        return self.drift_detected
