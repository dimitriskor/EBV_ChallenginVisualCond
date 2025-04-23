# Load data from utils.Events
# from info_data get sequence that is red
# split the data and keep only those of interest
#       Create dataloader? (maybe in next iteration of code)
# create slices of 50ms? of splitted data
# run on inference
# decode output


import matplotlib.pyplot as plt
from utils.Events import Events
import numpy as np
import os
# from utils.visualize import fetch_data_to_write, get_slices
from utils.utils import optimal_distance, postprocess, get_annotated_frame
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary
import torchvision
from RVT.config.modifier import dynamically_modify_train_config
from RVT.modules.utils.fetch import fetch_data_module, fetch_model_module
from pathlib import Path
from torch.backends import cuda, cudnn
import pickle
import cv2




folder_names = list(os.listdir('../../data/DSEC/test_events/'))
prefix = '../../data/DSEC/test_events/'
opt_distance = optimal_distance('DSEC')[0]


for folder in folder_names:

    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=10, height=480, width=640)
    slices = [[0, data.ms_to_idx.shape[0]//50]]
    data.save_ev2fr_as_images()

    del data
    
folder_names = list(os.listdir('../../data/DSEC/train_events/'))
prefix = '../../data/DSEC/train_events/'
opt_distance = optimal_distance('DSEC')[0]


for folder in folder_names:

    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=10, height=480, width=640)
    slices = [[0, data.ms_to_idx.shape[0]//50]]
    data.save_ev2fr_as_images()

    del data