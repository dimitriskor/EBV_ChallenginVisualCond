import torch
import norse
from utils import utils
import utils.YOLOModel as ym
from utils.Events import Events
import os
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

grid_size = 12

model = ym.YOLOBase(grid_size=grid_size, num_classes=9, bbox_per_cell=3)
model = norse.torch.Lift(model)
model.load_state_dict(torch.load('yolov1_model_v2_on_Glare_clean.pt', weights_only=True))
model.eval()
model = model.cpu()
folder_names = list(os.listdir('../../data/DSEC/train_events/'))
prefix = '../../data/DSEC/train_events/'



def run_inference(model, data, folder, num_classes=9):
    '''
        Run model on inference and create video with the output.
    '''
    print(type(data))
    output_video = f'../videos/network_res/{folder}.mp4'
    extra_channel = data[:, 0, :, :].unsqueeze(1)
    print(data.shape, 'before permute')
    data = torch.cat((data, extra_channel), dim=1).unsqueeze(1).float()
    print(data.shape, 'after permute')
    output = model(data)

    height, width = data[0].shape[2:4]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(output_video, fourcc, 100, (width, height))

    for idx, out in enumerate(output):
        frame = utils.postprocess_YOLO(out[0], data[idx][0], num_classes)
        plt.figure(figsize=(10, 10))
        frame = (frame/255)
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()




for folder in folder_names:
    # if '10_a' in folder or '04_f' in folder or '10_b' in folder or '09_b' in folder or '09_c' in folder:
    #     continue
    # if '00_b' not in folder:
    #     continue
    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=10, height=480, width=640)
    slices = [list(range(0, data.ms_to_idx.shape[0] // 50))]

    if slices == []:
        continue
    for s_id, slice in enumerate(slices):
        objects = [0, 0, 0, 0] # frames, class 0, ... 
        count = 0
        while len(slice) > 500:
            slice, slice_left = slice[:500], slice[500:] 
            events = torch.tensor(data.events_to_frames_DSEC(slice))
            run_inference(model, events, folder+str(count))
            slice = slice_left
            count += 1
        events = torch.tensor(data.events_to_frames_DSEC(slice))
        run_inference(model, events, folder+str(count))
    del data