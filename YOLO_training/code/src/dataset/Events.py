import os
import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import utils.entropy as entropy
from tqdm import tqdm


class Events():

    def __init__(self, filename, stack_events=5, height=480, width=640) -> None:
        self.filename = filename
        self.stack_events = stack_events
        self.height = height
        self.width = width
        self.file = h5py.File(filename, 'r')
        self.events = np.stack([np.array(self.file['events']['p'], dtype=np.uint16), np.array(self.file['events']['x'], dtype=np.uint16), np.array(self.file['events']['y'], dtype=np.uint16)]).T
        self.t = np.array(self.file['events']['t'])
        self.ms_to_idx = np.array(self.file['ms_to_idx'])
            
    def __del__(self):
        self.file.close()
        
    




    
    
    def events_to_frames_DSEC(self, slice=None, full=False):
        '''
            Transforms the event tuples (t, p, x, y) to a 4d tensor of shape (T, p, x, y), where contains non-overlapping bin events for durations self.stack_events ms
        '''
        print('Converting event tuples to array')
        self.events = np.stack([np.array(self.file['events']['p'], dtype=np.uint16), np.array(self.file['events']['x'], dtype=np.uint16), np.array(self.file['events']['y'], dtype=np.uint16)]).T
        self.t = np.array(self.file['events']['t'])

        if full:
            start = self.ms_to_idx[0]*50//self.stack_events
            stop = self.ms_to_idx[-1]*50//self.stack_events
            size = (stop-start, 2, self.height, self.width)
        else:
            start = slice[0]*50//self.stack_events
            stop = (slice[-1])*50//self.stack_events
            size = (stop-start, 2, self.height, self.width)
        as_frames = np.zeros(size, dtype=np.uint8)

        # If number of stack events is changed to self.stasck_events, it gets way slower
        for i in tqdm(range(stop-start)):
            subset =  self.events[self.ms_to_idx[slice[0]*50+self.stack_events*i]:self.ms_to_idx[slice[0]*50+self.stack_events*(i+1)]]
            p, x, y = subset[:, 0], subset[:, 1], subset[:, 2]
            np.add.at(as_frames[i], (p, y, x), 1)
            # print(np.max(as_frames[i]), np.min(as_frames[i]))
            # fv
        print('Done\n')
        return as_frames


    
    
    def events_to_frames_dataloader(self, slice=None, full=False):
        '''
            Transforms the event tuples (t, p, x, y) to a 4d tensor of shape (T, p, x, y), where contains non-overlapping bin events for durations self.stack_events ms
        '''
        if full:
            start = self.ms_to_idx[0]*self.stack_events
            stop = self.ms_to_idx[-1]*self.stack_events
            size = (stop-start, 2, self.height, self.width)
        else:
            start = slice[0]//self.stack_events
            stop = (slice[-1])//self.stack_events
            size = (stop-start, 2, self.height, self.width)
        self.events = np.stack([np.array(self.file['events']['p'][self.ms_to_idx[slice[0]]:self.ms_to_idx[slice[-1]]], dtype=np.uint16), 
                                np.array(self.file['events']['x'][self.ms_to_idx[slice[0]]:self.ms_to_idx[slice[-1]]], dtype=np.uint16), 
                                np.array(self.file['events']['y'][self.ms_to_idx[slice[0]]:self.ms_to_idx[slice[-1]]], dtype=np.uint16)]).T
        as_frames = np.zeros(size, dtype=np.uint8)
        for i in (range(stop-start)):
            subset =  self.events[self.ms_to_idx[slice[0] + self.stack_events*i]-self.ms_to_idx[slice[0]]:self.ms_to_idx[(slice[0]) + self.stack_events*(i+1)]-self.ms_to_idx[slice[0]]]
            p, x, y = subset[:, 0], subset[:, 1], subset[:, 2]
            np.add.at(as_frames[i], (p, y, x), 1)
        # print(np.max(as_frames), np.min(as_frames))
            # fv
        return as_frames

  


    def save_ev2fr_as_images(self):
        folder = self.filename.split('/')[-4]
        try:
            os.mkdir(f'../../data/DSEC/EBV_frames/{folder}')
        except Exception as e:
            print(e)
        fname = f'../../data/DSEC/EBV_frames/{folder}'
        slices = [np.linspace(0, self.ms_to_idx.shape[0]//50, self.ms_to_idx.shape[0]//50+1).astype(np.int32)]
        for s_id, slice in enumerate(slices):
            print(slice)
            while slice[-1]-slice[0] > 1000:
                slice, slice_left = slice[:1000], slice[999:] 
                events = self.events_to_frames_DSEC(slice)
                for f_idx, fr in tqdm(enumerate(events)):
                    fr = np.concatenate([fr, fr[:1]], axis=0)
                    fr = np.transpose(fr, (1,2,0))*4
                    plt.figure()
                    plt.imsave(fname+f'/{(slice[0]*50//self.stack_events+f_idx):05}.png', fr)
                    plt.close()
                slice = slice_left
            events = self.events_to_frames_DSEC(slice)
            for f_idx, fr in tqdm(enumerate(events)):
                fr = np.concatenate([fr, fr[:1]], axis=0)
                fr = np.transpose(fr, (1,2,0))*4
                plt.figure()
                plt.imsave(fname+f'/{(slice[0]*50//self.stack_events+f_idx):05}.png', fr)
                plt.close()
  
  
    def calculate_entropy(self, levels, entropy_fn=entropy.first_order_entropy, stack_events=None):
        '''
            It calculates the entropy under the entropy_fn for every frame of bin stack_events.
            levels: the probability space for the pixel values
            stack_events is given in ms
        '''
        if stack_events == None:
            stack_events = self.stack_events
        
        index = 0
        entropy = []
        while index + self.first_ts < self.last_ts:
            print(f'\r{index/(self.last_ts-self.first_ts)*100:0,.2f}/100', end='', sep='')
            frame = self.bin_events_to_frame(index, index + stack_events*1000)
            entropy.append(entropy_fn(frame, levels))
            index += stack_events*1000
        return np.array(entropy)    


