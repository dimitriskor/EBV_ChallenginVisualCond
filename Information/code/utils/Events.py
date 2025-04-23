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
        if 'DDD20' in filename:
            self.frame_ts = np.array(self.file['frame_ts'])
            self.events_as_array = np.array(self.file['event'])
            self.first_ts = self.file['event'][0][0]
            self.last_ts = np.max(self.events_as_array)
            self.is_last_ts_last_ev = self.last_ts == self.events_as_array[-1][0]
        elif 'DSEC' in filename:
            # self.events = np.stack([self.file['events']['t'], self.file['events']['p'], self.file['events']['x'], self.file['events']['y']]).T
            # self.events = np.stack([np.array(self.file['events']['p'], dtype=np.uint16), np.array(self.file['events']['x'], dtype=np.uint16), np.array(self.file['events']['y'], dtype=np.uint16)]).T
            # self.t = np.array(self.file['events']['t'])
            self.ms_to_idx = np.array(self.file['ms_to_idx'])
            
    def __del__(self):
        self.file.close()


    def events_to_frames(self, start_ts=0, end_ts=-1, view=False, stack_events=None, normalize = False):
        '''
            Stream events from file f.
            start_ts: starting timestep
            end_ts: Last timestep. Default -1 streams until the last event
            view: Visualize events as frames
            bin: Stack events for bin ms in a single frame 
        '''

        if stack_events == None:
            stack_events = self.stack_events
        
        if view:
            win_name = 'Events'
            cv2.namedWindow(win_name)

        events = self.file['event']
        first_ts = self.first_ts + start_ts*1000
        last_ts = self.last_ts if end_ts == -1 else min(self.first_ts + end_ts, self.last_ts)
        frame = np.zeros((self.height, self.width))
        images = []
        t_0 = first_ts
        n_frame=0
        print(first_ts, last_ts)
        st = time.time()
        for e in events:
            if e[0] < t_0 or e[0] > last_ts:
                continue
            if e[0] < t_0 + stack_events*1000:
                frame[e[2], e[1]] += 2*(e[3]-0.5) 
            else:
                t_0 += stack_events*1000
                if normalize == True:
                    frame = frame if frame.max() - frame.min() == 0 else (frame - frame.min()) / (frame.max() - frame.min())
                frame = (frame*255).astype(np.uint8)
                images.append(frame)
                if view:
                    sleep_time = max(0, 0.001*bin - time.time() + st)
                    time.sleep(sleep_time)
                    cv2.imshow(win_name, frame)
                    cv2.waitKey(1)
                    st = time.time()
                print('\r',n_frame, end='', sep='')
                n_frame += 1
        return images
    
        



    def bin_events_to_frame(self, start_ts, end_ts):
        '''
            Passes an numpy array of events and two timestamps from which it forms an frame from the events in between the two timestamps.
            It is used to calculate the entropy from frames of events and avoid calling the 'events_to_frames" (and constructing a huge 3d array)
            start_ts and end_ts are given in ms
        '''
        if not hasattr(self, 'last_index_access'):
            self.last_index_access = 0
        start_i = self.last_index_access
        while True:
            if self.events_as_array[start_i][0] >= self.first_ts + start_ts:
                break
            start_i += 1
        end_i = start_i  
        while True:
            if self.events_as_array[end_i+1][0] >= min(self.first_ts + end_ts, self.last_ts):
                break
            end_i += 1
        self.last_index_access = end_i
        events_slice = self.events_as_array[start_i:end_i]
        frame = np.zeros((self.height, self.width))
    
        # Extract event details
        y_coords = events_slice[:, 2]
        x_coords = events_slice[:, 1]
        polarities = events_slice[:, 3]
        
        # Vectorized accumulation of event values in the frame
        np.add.at(frame, (y_coords, x_coords), 2 * (polarities - 0.5))
        
        # Normalize the frame
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max != frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        
        # Scale to [0, 255] and convert to uint8
        frame = (frame * 255).astype(np.uint8)

        return frame
    
    
    
    
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

  

    '''
                self.events = np.stack([np.array(self.file['events']['p'][self.ms_to_idx[slice[0]]:self.ms_to_idx[slice[-1]]], dtype=np.uint16), 
                                    np.array(self.file['events']['x'][self.ms_to_idx[slice[0]]:self.ms_to_idx[slice[-1]]], dtype=np.uint16), 
                                    np.array(self.file['events']['y'][self.ms_to_idx[slice[0]]:self.ms_to_idx[slice[-1]]], dtype=np.uint16)]).T
    '''



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


