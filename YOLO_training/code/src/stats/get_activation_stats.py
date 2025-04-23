import pickle
from utils.Events import Events
import numpy as np
import os
from utils.visualize import fetch_data_to_write, get_slices
from utils.utils import optimal_distance, postprocess
import torch


stats = {}

folder_names = list(os.listdir('../../data/DSEC/train_events/'))
prefix = '../../data/DSEC/train_events/'
opt_distance = optimal_distance('DSEC')[0]

for folder in folder_names:
    # if '10_a' in folder or '04_f' in folder or '10_b' in folder or '09_b' in folder or '09_c' in folder:
    #     continue
    # if '04_f' not in folder:
        # continue
    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=1, height=480, width=640)
    metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  folder)
    metric = metric['partial_contrast']
    distance = np.sum(np.square((metric-opt_distance)), axis=1)
    slices = get_slices(distance)
    total_events = data.t.shape[0]
    clear_events = total_events
    glare_frames = 0
    if slices == []:
        glare_frames = 1
        ev_per_frame_glare = 0
    else: 
        for s_id, slice in enumerate(slices):
            start = slice[0]
            end = slice[-1]
            glare_frames += end - start + 1
            clear_events -= data.t[data.ms_to_idx[start]:data.ms_to_idx[end+1]].shape[0]
    glare_events = total_events - clear_events
    clear_frames = data.ms_to_idx.shape[0] - glare_frames
    ev_per_frame_clear = clear_events/clear_frames
    ev_per_frame_glare = glare_events/glare_frames
    print("Events per clear frame (50ms)", ev_per_frame_clear)
    print("Events per glared frame (50ms)", ev_per_frame_glare)
    d = {"Events per clear frame (50ms)": ev_per_frame_clear,
         "Events per glared frame (50ms)": ev_per_frame_glare}
    stats[folder] = d
    del data


folder_names = list(os.listdir('../../data/DSEC/test_events/'))
prefix = '../../data/DSEC/test_events/'
opt_distance = optimal_distance('DSEC')[0]

for folder in folder_names:
    # if '10_a' in folder or '04_f' in folder or '10_b' in folder or '09_b' in folder or '09_c' in folder:
    #     continue
    # if '04_f' not in folder:
        # continue
    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=1, height=480, width=640)
    metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  folder)
    metric = metric['partial_contrast']
    distance = np.sum(np.square((metric-opt_distance)), axis=1)
    slices = get_slices(distance)
    total_events = data.t.shape[0]
    clear_events = total_events
    glare_frames = 0
    if slices == []:
        glare_frames = 1
        ev_per_frame_glare = 0
    else: 
        for s_id, slice in enumerate(slices):
            start = slice[0]
            end = slice[-1]
            glare_frames += end - start + 1
            clear_events -= data.t[data.ms_to_idx[start]:data.ms_to_idx[end+1]].shape[0]
    glare_events = total_events - clear_events
    clear_frames = data.ms_to_idx.shape[0] - glare_frames
    ev_per_frame_clear = clear_events/clear_frames
    ev_per_frame_glare = glare_events/glare_frames
    print("Events per clear frame (50ms)", ev_per_frame_clear)
    print("Events per glared frame (50ms)", ev_per_frame_glare)
    d = {"Events per clear frame (50ms)": ev_per_frame_clear,
         "Events per glared frame (50ms)": ev_per_frame_glare}
    stats[folder] = d
    del data

print(stats)
with open('../statistics/events_per_frames.pkl', 'wb') as f:
    pickle.dump(stats, f)