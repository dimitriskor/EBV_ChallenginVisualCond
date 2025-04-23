### Create frames for EBV representations
#
# Each frame consists of 3 channels to allow compatibility with existing YOLO models for fine-tuning/training
#
# bin representation:
#   Each channel consists of binned events up to a given ms. The channels contain binned events of 5ms, 10ms and 20ms. Example: for frame at time 50ms, ch1 -> 45-50ms, ch2 -> 40-50ms, ch3 -> 30-50ms
#
# EROS representation:
#   Each channel contain an EROS implementation with different deltas. ch1: 0.25, ch2: 0.5, ch3: 0.75
#
# G-EROS representation:
#   Each channel contain an G-EROS implementation with different sigmas. ch1: 1, ch2: 2, ch3: 4
#
# Conv-EROS representation:
#   Each channel contain an Conv-EROS implementation with different kernel sizes. ch1: 3, ch2: 5, ch3: 7
#
# Sequential representation:
#   Each channel contain an SEQUENTIAL implementation with fixed ms integration time.
#
# Each frame is computed every 50ms to align with existing annotation and frame-based representation


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from src.representations.bins import BINS
from src.representations.EROS import EROS
from src.representations.GEROS import GEROS
from src.representations.sequential import SEQUENTIAL
import cv2



def create_figures(name, folder, frames):
    os.makedirs(f'../../representations/{folder}/{name}', exist_ok=True)
    for idx, frame in enumerate(frames):
        if frame.shape[2] == 3 or frame.shape[2] == 1:
            # plt.figure()
            # plt.imshow(frame)
            # plt.axis('off')
            # plt.gca().set_position([0, 0, 1, 1])
            # plt.savefig(f'../../representations/{folder}/{name}/{idx:05}.png', transparent=False)
            # plt.close()
            cv2.imwrite(f'../../representations/{folder}/{name}/{idx:05}.png', frame)
        else:
            torch.save(frame, f'../../representations/{folder}/{name}/{idx:05}.png')


names = ['zurich_city_08_a', 'zurich_city_01_a', 'interlaken_00_f']
names = os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/')

bins = BINS(bms=(5, 20, 80), clip_val= 8)
eros = EROS()
geros = GEROS()
seq = SEQUENTIAL(bin=10, delay=20, clip_val= 8)

for name in names:
    print(name)
    rec=f'/mnt/raid0a/Dimitris/DSEC/event_recordings/{name}/events/left/events.h5'
    # bin_events = bins.generate_frames(rec)
    # create_figures(name, 'bins_80_clip_8_times4', bin_events)
    eros_events = eros.generate_frames(rec)
    create_figures(name, 'EROS', eros_events)
    # geros_events = geros.generate_frames(rec)
    # create_figures(name, 'GEROS', geros_events)
    # seq_events = seq.generate_frames(rec)
    # create_figures(name, 'seq_10_delay_20_clip_8', seq_events)
