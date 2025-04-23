# sequential representation:
#   Each channel consists of binned events of given ms. The channels contain binned events of 5ms, 5ms and 5ms. Example: for frame at timems, ch1 -> 45-50ms, ch2 -> 40-45ms, ch3 -> 35-40ms
#   Delay sets a time to skip between integrations. Eg bin=5, delay=5: ch1 -> 45-50ms, ch2 -> 35-40ms, ch3 -> 25-30ms

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from src.dataset.Events import Events


class SEQUENTIAL():
    
    def __init__(self, bin, channels = 3, delay = 0, clip_val = None):
        self.bin = bin
        self.channels = channels
        self.delay = delay
        self.clip_val = clip_val

    def generate_frames(self, rec):
        events = Events(rec)
        ms_size = events.ms_to_idx.shape[0]//50
        size = (ms_size, events.height, events.width, self.channels)
        as_frames = np.zeros(size, dtype=np.uint8)
        for i in tqdm(range(1, ms_size)):
            for c in range(self.channels):
                subset =  events.events[events.ms_to_idx[i*50-(c+1)*self.bin-c*self.delay]:events.ms_to_idx[i*50-(c)*self.bin-c*self.delay]]
                x, y = subset[:, 1], subset[:, 2]
                np.add.at(as_frames[i], (y, x, c), 1)
            if self.clip_val != None:
                as_frames[i] = np.clip(as_frames[i], 0, self.clip_val-1)*int(256//self.clip_val)
        return as_frames
    
    def generate_single_frame(self, rec, ms, keep_polarity = True, augment = []):
        # events = Events(rec)
        if ms - self.bin*self.channels - self.delay*(self.channels-1) < 0:
            # print(ms - self.bin*self.channels - self.delay*(self.channels-1))
            return False
        self.file = h5py.File(rec, 'r')
        self.ms_to_idx = np.array(self.file['ms_to_idx'][ms - self.bin*self.channels - self.delay*(self.channels-1):ms])
        duration = self.bin*self.channels + self.delay*(self.channels-1)
        first_pos = self.ms_to_idx[0]
        last_pos = self.ms_to_idx[-1]
        self.events = np.stack([np.array(self.file['events']['p'][first_pos:last_pos], dtype=np.uint16), 
                                np.array(self.file['events']['x'][first_pos:last_pos], dtype=np.uint16), 
                                np.array(self.file['events']['y'][first_pos:last_pos], dtype=np.uint16)]).T

        height, width = 480, 640
        size = (height, width, self.channels, 2) if keep_polarity else (height, width, self.channels)
        frame = np.zeros(size, dtype=np.uint8)
        for c in range(self.channels):
            subset =  self.events[self.ms_to_idx[duration-(c+1)*self.bin-c*self.delay-1]-first_pos:self.ms_to_idx[duration-(c)*self.bin-c*self.delay-1]-first_pos]
            x, y, p = subset[:, 1], subset[:, 2], subset[:, 0]
            if 'jitter' in augment:
                x += np.random.choice([-1, 0, 1], size=x.shape[0], p=[0.25, 0.5, 0.25]).astype(np.uint16)
                y += np.random.choice([-1, 0, 1], size=y.shape[0], p=[0.25, 0.5, 0.25]).astype(np.uint16)
                x = np.clip(x, 0, 639)
                y = np.clip(y, 0, 479)
            if keep_polarity:
                np.add.at(frame, (y, x, c, p), 1)
                # frame = torch.sum(frame, dim=-1)
            else:
                np.add.at(frame, (y, x, c), 1)
        frame = frame.reshape(height, width, 2*self.channels) if keep_polarity else frame
        if 'noise' in augment:
            frame = add_salt_and_pepper_noise(frame)

        if self.clip_val != None:
            frame = np.clip(frame, 0, self.clip_val-1)*int(256//self.clip_val)
        return frame



def add_salt_and_pepper_noise(frame, noise_ratio=0.02):
    """
    Add salt-and-pepper noise to a frame.

    :param frame: NumPy array (grayscale or color image)
    :param noise_ratio: Probability of a pixel being affected (default: 2%)
    :return: Noisy frame
    """
    noisy_frame = frame.copy()
    
    # Generate random mask for salt and pepper
    noise_mask = np.random.choice([0, 1, 2], size=frame.shape, p=[noise_ratio / 2, 1 - noise_ratio, noise_ratio / 2])
    
    # Apply salt (white: 255) and pepper (black: 0) noise
    noisy_frame[noise_mask == 0] = 0    # Pepper
    noisy_frame[noise_mask == 2] += 1  # Salt

    return noisy_frame