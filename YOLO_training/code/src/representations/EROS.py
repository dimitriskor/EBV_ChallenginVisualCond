# EROS representation:
#   Each channel contain an EROS implementation with different deltas. ch1: 0.25, ch2: 0.5, ch3: 0.75

import enum
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from src.dataset.Events import Events
from copy import copy



class EROS():
    
    def __init__(self, kernel_size=7, deltas=[0.25, 0.5, 0.75]):
        self.kernel_size = kernel_size
        self.half_size = kernel_size // 2
        self.deltas = deltas
        for i, d in enumerate(self.deltas):
            self.deltas[i] = d ** (1/kernel_size)
        print(self.deltas)

    def generate_frames(self, rec):
        events = Events(rec)
        frames = []
        image = np.zeros((640, 480, len(self.deltas)))
        t0 = events.t[0]
        c = 0
        for i, e in tqdm(enumerate(events.events), total=events.events.shape[0]):
            x, y = e[1], e[2]
            t = events.t[i]
            x_start, x_end = max(x - self.half_size, 0), min(x + self.half_size + 1, image.shape[0])
            y_start, y_end = max(y - self.half_size, 0), min(y + self.half_size + 1, image.shape[1])

            # Extract and multiply
            for i, d in enumerate(self.deltas):
                image[x_start:x_end, y_start:y_end, i] *= d

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

                plt.imshow(frames[-1])
                plt.savefig('sdfgdb.png')
        return frames