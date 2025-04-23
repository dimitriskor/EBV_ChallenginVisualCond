# G-EROS representation:
#   Each channel contain an G-EROS implementation with different sigmas. ch1: 1, ch2: 2, ch3: 4

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from src.dataset.Events import Events
from copy import copy


def gaussian_kernel(size=7, sigma=1):
    """
    Generate a Gaussian kernel.

    Parameters:
    - size: Size of the kernel (e.g., 7 for a 7x7 kernel).
    - sigma: Standard deviation of the Gaussian distribution.

    Returns:
    - A 2D NumPy array representing the Gaussian kernel.
    """
    # Define the grid range
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    return kernel




class GEROS():
    
    def __init__(self, kernel_size=9, sigmas=(1, 2, 4)):
        self.kernel_size = kernel_size
        self.half_size = kernel_size // 2
        self.sigmas = sigmas

    def generate_frames(self, rec):
        events = Events(rec)
        frames = []
        image = np.zeros((640, 480, len(self.sigmas)))
        kernel = np.zeros((self.kernel_size, self.kernel_size, len(self.sigmas)))
        for i, s in enumerate(self.sigmas):
            kernel[..., i] = gaussian_kernel(self.kernel_size, s)
        t0 = events.t[0]
        c = 0
        for i, e in tqdm(enumerate(events.events), total=events.events.shape[0]):
            x, y = e[1], e[2]
            t = events.t[i]
            x_start, x_end = max(x - self.half_size, 0), min(x + self.half_size + 1, image.shape[0])
            y_start, y_end = max(y - self.half_size, 0), min(y + self.half_size + 1, image.shape[1])

            kernel_x_start = max(self.half_size - x, 0)
            kernel_x_end = kernel_x_start + (x_end - x_start)

            kernel_y_start = max(self.half_size - y, 0)
            kernel_y_end = kernel_y_start + (y_end - y_start)

            # Extract and multiply
            for i, s in enumerate(self.sigmas):
                image[x_start:x_end, y_start:y_end, i] *= kernel[kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end, i]

            # Ensure the center point is set
            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                image[x, y] = 1

            if t-t0 >= 50000:
                # print(t-t0)
                t0 += 50000
                frames.append(copy(np.transpose(image, (1, 0, 2))))
                c += 1
            # if c == 10:
            #     break

        return frames