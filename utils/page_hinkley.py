# utils/page_hinkley.py
import numpy as np

class PageHinkley:
    

    def __init__(self, delta=1.0, lambd=500, min_instances=100, verbose=False):
        
        self.delta = float(delta)
        self.lambd = float(lambd)
        self.min_instances = int(min_instances)
        self.verbose = verbose
        self.reset()

    def reset(self):
       
        self.sum = 0.0
        self.x_mean = 0.0
        self.num_instances = 0
        self.min_sum = 0.0
        self.drift_detected = False

    def update(self, error):
        
        try:
            x = float(error)
        except Exception:
           
            return False

       
        self.num_instances += 1

       
        old_mean = self.x_mean
        self.x_mean += (x - self.x_mean) / self.num_instances

       
        self.sum += x - (old_mean + self.delta)

        
        if self.sum < self.min_sum:
            self.min_sum = self.sum

        if self.verbose:
            print(f"[PageHinkley] n={self.num_instances}, x={x:.6f}, mean={self.x_mean:.6f}, "
                  f"sum={self.sum:.6f}, min_sum={self.min_sum:.6f}, lambd={self.lambd}")

        
        if self.num_instances < self.min_instances:
            return False

       
        self.drift_detected = (self.sum - self.min_sum) > self.lambd
        return self.drift_detected
