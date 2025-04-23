
from logging import raiseExceptions
import pickle
import numpy as np
from ultralytics import YOLO
import norse
import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from Information.code.ultralytics.ultralytics.data import augment
from dataset.Events import Events
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import utils.entropy as entropy
from tqdm import tqdm
from utils.utils import optimal_distance
from utils.data import fetch_data_to_write, remap
from src.representations.bins import BINS
from src.utils.augmentations import zoom_translate_crop, augment_video_with_flip_and_rotation, silence_channels
from src.representations.sequential import SEQUENTIAL
import src.dataset.visual_config as visual_config
import random


"""
    Change to support functionality for night, glare, dawn dataset separation and channel and encoding schemes differences
    Class init signature:
        file_list: list of files for dataset
        datasplit: split of data. If not [] or filelist == None then file_list is overwritten. Allowed values: night, glare, dawn, random
                    -- random: random select files from all files
                    -- night: removes night data from training set
                    -- dawn: removes dawn and night data from training set (exposure > 10ms).
                    -- glare: removes glare data from training set. Glare is identified using detection algorithm. It can be independently applied
                    
                If more arguments are passed ['dawn', 'glare'] then the val test holds from both datasets splits
                The test set is always the hold out set from visual_config
                The train/test split ratio is fixed to 2/1.

            Function to load the ids as a list and split them into train/val for the images to fetch needed 

        trainset: Boolean that creates the dataset based on the datasplit. If True, keeps the clean recordings. Used only if datasplit.
        input_channels: Number on bins in input
#DONE   channel_repr: representation format of channels. Options 'sequential'/'static'/'static_inv'. 
                    -- sequestial: bins events for Xms for each channel sequentially (Eg. ch1: 0-5ms, ch2: 5-10ms, ...)
                    -- bins: bins events from reference ms backwards (Eg. ch1: 45-50ms, ch2: 40-50ms, ch3: 30-50ms, ...)
                    -- ### Not implemented ### static_inv: bins events from reference ms backwards (Eg. ch1: 50-55ms, ch2: 50-60ms, ch3: 50-70ms, ...)
        bins: int/list that dictates the integration of each channel. Int only if channel_repr == 'sequential'

"""



class ReprLoader(Dataset):
    def __init__(self, 
                 file_list, 
                 datasplit, 
                 trainset, 
                 input_channels, 
                 channel_repr, 
                 bins, 
                 static_thres=100, 
                 event_static=True, 
                 rec_prefix='/mnt/raid0a/Dimitris/DSEC/event_recordings/', 
                 obj_prefix='../../labels/11x/0.2/',
                 classes = {'person' : 0, 'bicycle' : 1, 'car' : 2, 'motorcycle' : 3, 'bus' : 5, 'truck' : 7, 'traffic light' : 9, 'stop sign' : 11},
                 relaxation = 0.02,
                 augmentations = []
                 ):
        
        self.file_list = file_list
        self.datasplit = datasplit
        self.input_channels = input_channels
        self.bins = bins
        self.channel_repr = channel_repr
        self.num_sequences_per_file = []
        self.rec_prefix = rec_prefix
        self.obj_prefix = obj_prefix
        self.max_bboxes = 20
        self.static_thres = static_thres
        self.event_static = event_static        # Not yet implemented
        self.optimal_distance = optimal_distance('DSEC')[0]
        self.classes = classes
        self.relaxation = relaxation
        self.night = visual_config.night
        self.dawn = visual_config.dawn_street_lights_on + visual_config.car_lights_on
        self.clear = visual_config.clear
        self.hold_out = visual_config.hold_out
        self.trainset = trainset
        self.augmentations = augmentations
        if self.trainset == False:
            self.file_list = self.hold_out
        else:
            if self.file_list == None or self.file_list == []:
                self.file_list = []
                if self.datasplit == ['random']:
                    self.file_list = self.night + self.dawn + self.clear
                    random.shuffle(self.file_list)
                    self.file_list = self.file_list[:2*len(self.hold_out)]
                if 'dawn' in self.datasplit:
                    self.file_list = self.clear
                    random.shuffle(self.file_list)
                    self.file_list = self.file_list[:2*len(self.hold_out)]
                if 'night' in self.datasplit:
                    self.file_list = self.dawn + self.clear
                    random.shuffle(self.file_list)
                    self.file_list = self.file_list[:2*len(self.hold_out)]
        self._calculate_num_sequences()


        # Use of exposure timestamp as well!!!!
        

    def _calculate_num_sequences(self):
        """
        Determine the number of sequences in each file without loading everything into memory.
        """
        for file_idx in self.file_list:
            file_path = self.rec_prefix + file_idx + '/events/left/events.h5'
            f = h5py.File(file_path, 'r')
            ms_to_idx = np.array(f['ms_to_idx'])
            f.close()
            self.num_sequences_per_file.append(int(len(ms_to_idx)//50))

    def __len__(self):
        """
        Total number of sequences across all files.
        """
        return sum(self.num_sequences_per_file)
    
    def __getitem__(self, idx):
        cumulative_sequences = np.cumsum(self.num_sequences_per_file)
        file_idx = np.searchsorted(cumulative_sequences, idx, side='right')
        if file_idx == 0:
            sequence_idx = idx
        else:
            sequence_idx = idx - cumulative_sequences[file_idx - 1]
        file_path = self.rec_prefix + self.file_list[file_idx] + '/events/left/events.h5'
        target_path = self.obj_prefix + self.file_list[file_idx] + f'.pkl' 
        # events = Events(file_path, stack_events=self.time_window_ms)
        start_idx = sequence_idx * 50
        if self.channel_repr == 'bins':
            bins = BINS(bms=self.bins, clip_val= 8)
            frame = bins.generate_single_frame(file_path, ms=start_idx, keep_polarity = False, augment=self.augmentations)
        if self.channel_repr == 'seq':
            seq = SEQUENTIAL(bin=20, delay=10, channels=5, clip_val= 16)
            frame = seq.generate_single_frame(file_path, ms=start_idx, keep_polarity = False, augment=self.augmentations)
        if frame is False:
            return 0, 0, False, False
        with open(target_path, 'rb') as f:
            target = pickle.load(f)
        target = target[(start_idx)//50]
        target = remap(target, self.file_list[file_idx], self.rec_prefix)[0]
        try:
            target_cx = (target[:, 0] + target[:, 2])/2
            target_w = torch.abs((target[:, 0] - target[:, 2]))
            target_cy = (target[:, 1] + target[:, 3])/2
            target_h = torch.abs(target[:, 1] - target[:, 3])
            target[:, 0] = target_cx/640
            target[:, 1] = target_cy/480
            target[:, 2] = target_w/640
            target[:, 3] = target_h/480
            target = target[target[:, 3] > self.relaxation]
            target = target[target[:, 2] > self.relaxation]
            target = target[:, [5, 0, 1, 2, 3, 4]]
            target = target[:, :5]
            target = target[target[:, 0] == self.classes['car'] or target[:, 0] == self.classes['bus'] or target[:, 0] == self.classes['truck']]
            target[:, 0] = 0
        except:
            pass
        metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  self.file_list[file_idx])
        metric = metric['partial_contrast']
        distance = np.sum(np.square((metric-self.optimal_distance)), axis=1)
        flag = False
        if distance[(start_idx)//50] > 0.025 and 'glare' in self.datasplit:
            flag = True
        # # del events
        if 'scale_trans' in self.augmentations:
            frame, target = zoom_translate_crop(frame, target, self.relaxation)
        if 'flip_rot' in self.augmentations:
            frame, target = augment_video_with_flip_and_rotation(frame, target)
        # if 'silence_channels' in self.augmentations:
        #     frame = silence_channels(frame)
        return frame, target, flag, True











class HDF5Dataset_YOLO(Dataset):
    def __init__(self, file_list, time_window_ms, sequence_length, static_thres=100, event_static=True, rec_prefix='/mnt/raid0a/Dimitris/DSEC/event_recordings/', obj_prefix='/mnt/raid0a/Dimitris/DSEC/object_detection_yolo/'):
        self.file_list = file_list
        self.time_window_ms = time_window_ms
        self.sequence_length = sequence_length
        self.num_sequences_per_file = []
        self.rec_prefix = rec_prefix
        self.obj_prefix = obj_prefix
        self.max_bboxes = 20
        self.static_thres = static_thres
        self.event_static = event_static
        self._calculate_num_sequences()
        self.optimal_distance = optimal_distance('DSEC')[0]

    def _calculate_num_sequences(self):
        """
        Determine the number of sequences in each file without loading everything into memory.
        """
        for file_idx in self.file_list:
            file_path = self.rec_prefix + file_idx + '/events/left/events.h5'
            f = h5py.File(file_path, 'r')
            ms_to_idx = np.array(f['ms_to_idx'])
            f.close()
            self.num_sequences_per_file.append(int(len(ms_to_idx)//self.time_window_ms//self.sequence_length))

    def __len__(self):
        """
        Total number of sequences across all files.
        """
        return sum(self.num_sequences_per_file)
    
    def __getitem__(self, idx):
        cumulative_sequences = np.cumsum(self.num_sequences_per_file)
        file_idx = np.searchsorted(cumulative_sequences, idx, side='right')
        if file_idx == 0:
            sequence_idx = idx
        else:
            sequence_idx = idx - cumulative_sequences[file_idx - 1]
        file_path = self.rec_prefix + self.file_list[file_idx] + '/events/left/events.h5'
        target_path = self.obj_prefix + self.file_list[file_idx] + f'.pkl' 
        events = Events(file_path, stack_events=self.time_window_ms)
        start_idx = sequence_idx * self.sequence_length * self.time_window_ms
        frames = events.events_to_frames_dataloader(slice=[start_idx, start_idx+self.sequence_length * self.time_window_ms])      
        with open(target_path, 'rb') as f:
            target = pickle.load(f)
        target = target[(start_idx+self.sequence_length * self.time_window_ms//2)//50:(start_idx+self.sequence_length * self.time_window_ms)//50]
        target = [remap(t, self.file_list[file_idx], self.rec_prefix)[0] for t in target]
        new_target = []
        for t in target:
            if t.shape[0] < self.max_bboxes:
                padding = torch.zeros((self.max_bboxes - t.shape[0], 6))  # 6 because we have [x, y, w, h, obj_conf, class_id]
                t = torch.cat([t, padding], dim=0)
                new_target.append(t)
        target = torch.stack(new_target)
        try:
            target_cx = (target[:,:, 0] + target[:,:, 2])/2
            target_w = torch.abs((target[:,:, 0] - target[:,:, 2]))
            target_cy = (target[:,:, 1] + target[:,:, 3])/2
            target_h = torch.abs(target[:,:, 1] - target[:,:, 3])
            target[:,:, 0] = target_cx
            target[:,:, 1] = target_cy
            target[:,:, 2] = target_w
            target[:,:, 3] = target_h
        except:
            target = torch.zeros((self.sequence_length * self.time_window_ms//50//2, 20, 6))
        metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  self.file_list[file_idx])
        metric = metric['partial_contrast']
        distance = np.sum(np.square((metric-self.optimal_distance)), axis=1)
        frames = torch.tensor(frames)
        flag = False
        if self.event_static:
            mid = frames.shape[0]//2
            if torch.sum(frames[mid:mid+self.time_window_ms]) < self.static_thres and torch.sum(target[0]) > 0:
                flag = True
        else:
            raise Exception("Not yet implemented. Use self.event_static=True. If frames can be loaded then fix")
        if distance[(start_idx+self.sequence_length * self.time_window_ms)//50] > 0.025:
            flag = True
        if len(frames) < self.sequence_length:
            raise ValueError(f"Not enough frames in {file_path}")
        # frames = torch.nn.functional.interpolate(frames, size=(640, 640), mode='bilinear', align_corners=False)
        frames = frames.permute(0, 2, 3, 1)
        del events
        return frames, target, flag


