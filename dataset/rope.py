import numpy as np

class RoPE:
    def __init__(self, d_model, theta_base=10000.0):
        self.d_model = d_model
        self.theta_base = theta_base
        self.freqs = 1.0 / (self.theta_base ** (np.arange(0, d_model, 2) / d_model))
        
    def rotate(self, x, position):
        seq_len, d_model = x.shape
        angles = position * self.freqs
        cos = np.cos(angles)[None, :]
        sin = np.sin(angles)[None, :]
        
        x_rotated = x.copy()
        x_rotated[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
        x_rotated[:, 1::2] = x[:, 1::2] * cos + x[:, 0::2] * sin
        return x_rotated
