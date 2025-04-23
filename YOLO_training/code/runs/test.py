# import matplotlib.pyplot as plt
# import imageio

# image = imageio.imread('../../representations/EROS/zurich_city_09_d/00001.png')
# print(image)
# plt.imshow(image[..., 2])
# plt.savefig('a.png')



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
#
# Each frame is computed every 50ms to align with existing annotation and frame-based representation


# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from tqdm import tqdm
# from src.representations.bins import BINS
# from src.representations.EROS import EROS
# from src.representations.GEROS import GEROS
# from src.representations.ConvEROS import ConvEROS



# def create_figures(name, folder, frames):
#     os.makedirs(f'../../representations/{folder}/{name}', exist_ok=True)
#     for idx, frame in enumerate(frames):
#         plt.figure()
#         plt.imshow(frame)
#         plt.axis('off')
#         plt.gca().set_position([0, 0, 1, 1])
#         plt.savefig(f'../../representations/{folder}/{name}/{idx:05}.png')
#         plt.close()


# names = ['zurich_city_08_a', 'zurich_city_01_a', 'interlaken_00_f']
# names = os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/')

# bins = BINS()
# eros = EROS()
# geros = GEROS()
# ceros = ConvEROS()

# for name in names[1:]:
#     print(name)
#     rec=f'/mnt/raid0a/Dimitris/DSEC/event_recordings/{name}/events/left/events.h5'
#     converos_events = ceros.generate_frames(rec)
#     g = converos_events[..., 1]
#     create_figures(name, 'ConvEROS', g)


# import cv2
# import matplotlib.pyplot as plt
# image = plt.imread('./test.png')
# print(image.shape, image)
# image[200,200,0:] = 1
# plt.axis('off')
# plt.imshow(image[190:210,190:210,0:])
# plt.savefig('sub_kernel.png')

# image = plt.imread('./test.png')
# image[202,202,0:] = 1
# plt.axis('off')
# plt.imshow(image[192:212,192:212,0:])
# plt.savefig('sub_kernel_2.png')


# import matplotlib.pyplot as plt
# import imageio
# import numpy as np
# img = plt.imread('../../representations/bins_80_clip_8/zurich_city_09_d/00002.png')
# print(img)
# plt.imshow(img)
# plt.savefig('csvdfgfdfgbf.png')



# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import h5py
# import hdf5plugin
# import numpy as np


# filename = '/mnt/raid0a/Dimitris/DSEC/event_recordings/zurich_city_04_b/events/left/events.h5'
# file = h5py.File(filename, 'r')
# ms_to_idx = np.array(file['ms_to_idx'][10000])
# f_p = file['events']['p'][:ms_to_idx]
# f_y = file['events']['y'][:ms_to_idx]
# f_x = file['events']['x'][:ms_to_idx]
# f_t = file['events']['t'][:ms_to_idx]
# # print(file['ms_to_idx'][-1]//1000000)

# with h5py.File('zurich_city_04_b.h5', 'w') as filewrite:
#     filewrite.create_dataset('p', data=f_p, compression="gzip", compression_opts=9)
#     filewrite.create_dataset('y', data=f_y, compression="gzip", compression_opts=9)
#     filewrite.create_dataset('x', data=f_x, compression="gzip", compression_opts=9)
#     filewrite.create_dataset('t', data=f_t, compression="gzip", compression_opts=9)



# from ultralytics import YOLO
# import cv2
# import numpy as np

# import matplotlib.pyplot as plt

# rgb_path = '/mnt/raid0a/Dimitris/DSEC/images_recordings/interlaken_00_a/images/left/rectified/000010.png'
# model = YOLO('yolo11m.pt')
# img = cv2.imread(rgb_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # results = model(img)
# # save_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
# # cv2.imwrite('aaaaaaaa.png', save_rgb)

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import h5py
# import hdf5plugin
# import numpy as np
# import cv2

# filename = '/mnt/raid0a/Dimitris/DSEC/event_recordings/interlaken_00_a/events/left/events.h5'
# file = h5py.File(filename, 'r')
# ms_to_idx = np.array(file['ms_to_idx'])


# events = np.stack([np.array(file['events']['p'][ms_to_idx[490]:ms_to_idx[500]], dtype=np.uint16), 
#                                 np.array(file['events']['x'][ms_to_idx[490]:ms_to_idx[500]], dtype=np.uint16), 
#                                 np.array(file['events']['y'][ms_to_idx[490]:ms_to_idx[500]], dtype=np.uint16)]).T

# print(events[:, 0].sum())
# frame = np.zeros((480, 640))
# p, x, y = events[:, 0], events[:, 1], events[:, 2]
# np.add.at(frame, (y, x), 1)
# cv2.imwrite('aaaaaaaaa.png', 128*frame)


# import pickle
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.utils.data import fetch_data_to_write, remap
# import torch

# target_path = '/mnt/raid0a/Dimitris/DSEC/object_detection_yolo/interlaken_00_a.pkl'

# with open(target_path, 'rb') as f:
#     target = pickle.load(f)
# target = target[10]
# print(target)
# target = remap(target, 'interlaken_00_a', '/mnt/raid0a/Dimitris/DSEC/event_recordings/')[0]
# print(target)

# # target_cx = (target[:, 0] + target[:, 2])/2
# # target_w = torch.abs((target[:, 0] - target[:, 2]))
# # target_cy = (target[:, 1] + target[:, 3])/2
# # target_h = torch.abs(target[:, 1] - target[:, 3])
# # target[:, 0] = target_cx/640
# # target[:, 1] = target_cy/480
# # target[:, 2] = target_w/640
# # target[:, 3] = target_h/480
# target = target[0, :4]

# x1, y1, x2, y2 = target
# x1 = int(x1)
# y1 = int(y1)
# x2 = int(x2)
# y2 = int(y2)
# frame = frame*128
# print( (x1, y1), (x2, y2),)
# cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)
# # cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)

# cv2.imwrite('avaaaaaaaaa.pnga.jpg', frame)


from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO
# from ultralytics.utils.metrics import fitness
from ultralytics.utils.torch_utils import de_parallel
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.loader import ReprLoader
# import albumentations as A
import numpy as np
import cv2



# **Parse Arguments**
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='Config file for dataset')
parser.add_argument('-d', '--device_id', type=int, default=0, help='Id of CUDA to use')
parser.add_argument('-s', '--model_size', type=str, default='m', help='Model size. Options are n, s, m, l, x')
args = parser.parse_args()


def proc(labels):
    dict = {'cls' : [],
            'bboxes': [],
            'batch_idx': []}
    for idx, b in enumerate(labels):
        for l in b:
            # print(l)
            dict['cls'].append(torch.tensor([0.]))
            dict['bboxes'].append(torch.tensor([l[1], l[2], l[3], l[4]]))
            dict['batch_idx'].append(idx)
    if dict['cls']:  
        dict['cls'] = torch.stack(dict['cls'])
    else:
        dict['cls'] = dict['cls'] = torch.empty(0, 1) 
    if dict['bboxes']:  
        dict['bboxes'] = torch.stack(dict['bboxes'])
    else:
        dict['bboxes'] = torch.empty(0, 4)

    if dict['batch_idx']:  
        dict['batch_idx'] = torch.tensor(dict['batch_idx'])
    else:
        dict['batch_idx'] = torch.empty(0, dtype=torch.long)



    # if dict['cls'].numel() == 0:
    #     dict['cls'] = torch.tensor([[]])
    #     print('here')
    # if dict['bboxes'].numel() == 0:
    #     print('here')
    #     dict['bboxes'] = torch.tensor([[]])
    # if dict['batch_idx'].numel() == 0:
    #     print('here')
    #     dict['batch_idx'] = torch.tensor([])
    # print(dict)
    return dict


# **Set Device**
device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
device = 'cpu'
# **Load YOLO Model**
yolo_model = YOLO("yolo11n").to(device)
criterion = yolo_model.loss


# **Extract Hyperparameters from YOLO**
# hyp = yolo_model.overrides
hyp = {'batch': 1,
       'lr0': 0.01,
       'lrf': 0.01,
       'weight_decay': 0.0005,
       'epochs': 40
       }


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized target tensors.
    """
    images, targets, glare_flag, input_flag = zip(*batch)  # Separate images and targets

    # Stack images (assumes images are tensors of the same shape)
    try:
        images = torch.tensor(np.array(images))
    except:
        # print(len(images))
        input_flag = [0]
    # Targets remain as a list of tensors (each tensor contains ground truth for one image)
    return images, targets, glare_flag, input_flag

# **Attach Preprocessing Module Before YOLO**
class CustomYOLOPipeline(nn.Module):
    def __init__(self, yolo):
        super(CustomYOLOPipeline, self).__init__()
        # self.preprocess_layer = TrainablePreprocessing(in_channels=10, out_channels=3)
        self.yolo = de_parallel(yolo.model)  # Remove potential DataParallel wrapper
        self.stride = getattr(self.yolo, 'stride', 32)  # Default to 32 if not found
        self.names = getattr(self.yolo, 'names', {'car': 2})

    def forward(self, x, *args, **kwargs):
        # x = self.preprocess_layer(x)  # Apply trainable preprocessing
        return self.yolo(x, *args, **kwargs)  # Pass to YOLO
    def fuse(self, verbose=False):
        """Required by Ultralytics YOLO during validation/inference."""
        if verbose:
            print("Fusion is not needed for CustomYOLOPipeline. Returning self.")
        return self  # Return self as a no-op (no fusing needed)

# **Replace YOLO's Backbone**
yolo_model.model = CustomYOLOPipeline(yolo_model)


# **Create Train and Validation DataLoader**
train_dataset = ReprLoader([], ['night'], True, 4, 'seq', (5, 10, 20, 40, 80), classes={'car' : 2}, augmentations=['jitter', 'noise', 'scale_trans', 'flip_rot', 'silence_channels'])
val_dataset = ReprLoader([], ['night'], False, 4, 'seq', (5, 10, 20, 40, 80), classes={'car' : 2})
# print(val_dataset.__getitem__(1))
# train_dataset.__getitem__(2)
train_loader = DataLoader(train_dataset, batch_size=hyp['batch'], shuffle=True, collate_fn=collate_fn, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=hyp['batch'], shuffle=False, collate_fn=collate_fn, num_workers=16)

# **Create Optimizer and Scheduler**
optimizer = optim.Adam(yolo_model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyp['lrf'], total_steps=hyp['epochs'])

# **Define Loss Function (Same as YOLO)**

# **Training Loop**
epochs = hyp['epochs']
for epoch in range(epochs):
    yolo_model.model.train()
    total_loss = 0

    for batch_idx, (inputs, labels, glare_flag, input_flag) in enumerate(train_loader):
        # break
        if sum(glare_flag) or (sum(input_flag)-hyp['batch']):
            continue
        inputs, labels = inputs.to(device), [t.to(device) for t in labels]  
        optimizer.zero_grad()

        # Forward pass (Preprocessing + YOLO)
        inputs = inputs.permute(0,3,1,2).float()
        inputs = F.pad(inputs, (0, 0, 80, 80))

        preds = yolo_model.yolo(inputs)

        # Compute YOLO loss
        labels = proc(labels)
        # print(labels)
        loss, loss_items = criterion(labels, preds)
        loss.backward()
        optimizer.step()

        # Track loss components
        box_loss, class_loss, dfl_loss = loss_items[:3]
        total_loss += loss.item()

        # **Print Loss in YOLO Format**
        print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | "
              f"box_loss: {box_loss:.4f}, class_loss: {class_loss:.4f}, dfl_loss: {dfl_loss:.4f}", end='\r')



