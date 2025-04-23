import os
from PIL import Image
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import src.utils.entropy as entropy
from tqdm import tqdm

class Frames:
    
    def __init__(self, filename, height=260, width=346) -> None:
        self.filename = filename
        self.height = height
        self.width = width
        if 'DDD20' in filename:
            self.file = h5py.File(filename, 'r')
            self.frames_as_array = np.array(self.file['frame'])
            self.timestamps_as_array = np.array(self.file['frame'])
            self.first_ts = self.file['frame_ts'][0]
            self.last_ts = np.max(self.timestamps_as_array)
            self.is_last_ts_last_ev = self.last_ts == self.timestamps_as_array[-1]
        if 'DSEC' in filename:
            self.frames_as_array = self.convert_images_to_array()


    def convert_images_to_array(self):
        prefix = f'{self.filename}/images/left/rectified/'
        frames = []
        print(f'Reading {self.filename.split("/")[-1]} images')
        frame_names = list(os.listdir(prefix))
        frame_names.sort()
        for image_name in tqdm(frame_names):
            image = cv2.imread(prefix+image_name)
            frames.append(image.copy())
            # image.close()
        print(f'Converting images to numpy array')

        return np.array(frames)



    def stream_frames(f, start_ts=0, end_ts=-1, view=True, height=260, width=346):
        '''
            Stream frames from file f. Timesteps will be reset to 0
            start_ts: starting timestep
            end_ts: Last timestep. Default -1 streams until the last frame
            view: Visualize events as frames
        '''
        if view:
            win_name = 'Frames'
            cv2.namedWindow(win_name)


        file = h5py.File(f, 'r')
        frames = np.array(file['frame'])
        '''

            ................. to be continued


        '''


    def calculate_entropy(self, levels, entropy_fn=entropy.first_order_entropy, ranges=None):
        '''
            It calculates the entropy under the entropy_fn for every frame.
            levels: the probability space for the pixel values
        '''
        entropies = []
        print('Calculating entropy')
        if ranges is not None and entropy_fn == entropy.partial_contrast:
            for image in tqdm(self.frames_as_array, total=self.frames_as_array.shape[0]):
                entropies.append(entropy_fn(image, levels, ranges=ranges))
        else:
            for image in tqdm(self.frames_as_array, total=self.frames_as_array.shape[0]):
                entropies.append(entropy_fn(image, levels))
        return np.array(entropies)    
