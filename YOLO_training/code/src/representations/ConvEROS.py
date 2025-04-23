# G-EROS representation:
#   Each channel contain an G-EROS implementation with different sigmas. ch1: 1, ch2: 2, ch3: 4

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from src.dataset.Events import Events
from copy import copy

class ConvEROS():
    
    def __init__(self, kernel_size=9, deltas=(0.25, 0.5, 0.75)):
        self.kernel_size = kernel_size
        self.half_size = kernel_size // 2
        self.deltas = deltas

    def generate_frames(self, rec):
        events = Events(rec)
        frames = []
        image = np.zeros((640, 480, len(self.deltas)))
        t0 = events.t[0]
        kernel = np.zeros((self.kernel_size, self.kernel_size, len(self.deltas)))
        for i, d in enumerate(self.deltas):
            kernel[..., i] = d
        
        c = 0
        with torch.no_grad():
            for i, e in tqdm(enumerate(events.events), total=events.events.shape[0]):
                x, y = e[1], e[2]
                t = events.t[i]
                conv_kernel = copy(kernel)
                padded_image = torch.tensor(np.pad(image, ((2*self.half_size, 2*self.half_size), 
                                                        (2*self.half_size, 2*self.half_size), (0, 0)), mode='constant', constant_values=0))
                # Apply the kernel to the local patch
                for i, d in enumerate(self.deltas):
                    local_patch = padded_image[x+self.half_size:x+3*self.half_size+1, y+self.half_size:y+3*self.half_size+1, i]
                    patch = padded_image[x:x+2*self.kernel_size-1, y:y+2*self.kernel_size-1, i]
                    patch = torch.tensor(patch).unsqueeze(0).unsqueeze(0)
                    conv = torch.tensor(local_patch).unsqueeze(0).unsqueeze(0)
                    result = F.conv2d(patch, conv, padding=0)
                    # print(local_patch.shape, patch.shape, result.shape, x, y)
                    padded_image[x+self.half_size:x+3*self.half_size+1, y+self.half_size:y+3*self.half_size+1, i] *= (d+result[0][0])
                    image = padded_image[self.half_size:-self.half_size, self.half_size:-self.half_size]


                # Ensure the center point is set
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    image[x, y] = 1

                if t-t0 >= 10:
                    print(t-t0)
                    t0 += 10
                    frames.append(copy(np.transpose(image, (1, 0, 2))))
                    plt.imshow(image[:,:, 0].T, cmap='gray')
                    print(conv_kernel[:,:,0])
                    plt.axis('off')
                    plt.gca().set_position([0, 0, 1, 1])
                    plt.savefig('sesdv.png')
                    plt.close()
                    c += 1
                if c == 100:
                    break

            
        return frames