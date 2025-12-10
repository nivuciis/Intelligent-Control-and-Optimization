# utils.py
import numpy as np

class Metrics:
    def __init__(self, expected, observed):
        self.expected = np.array(expected)
        self.observed = np.array(observed)
        self.errors = self.expected - self.observed
        self.steps = np.arange(1, len(self.errors) + 1)

    def mse(self):
        return np.cumsum(self.errors ** 2) / self.steps

    def goodhart(self, mse_val):
        e1 = np.cumsum(self.observed) / self.steps
        e2 = np.cumsum(self.observed - e1) / self.steps
        c1, c2, c3 = 0.23, 0.23, 0.54
        return (e1 * c1) + (e2 * c2) + (mse_val * c3)
    def itae(self):
        return np.cumsum(self.steps * np.abs(self.errors)) / self.steps
    def iae(self):
        return np.cumsum(np.abs(self.errors)) / self.steps
    def ise(self):
        return np.cumsum(np.abs(self.errors) ** 0.5) / self.steps

def get_ref(t, type='step'):
    """ Simple reference function"""
    if type == 'step':
        return 5.0 * np.ones_like(t) if not np.isscalar(t) else 5.0
    elif type == 'ramp':
        return 0.5 * t if not np.isscalar(t) else 0.5 * t
    elif type == 'sine':
        return 5.0 * np.sin(0.5 * t) if not np.isscalar(t) else 5.0 * np.sin(0.5 * t)