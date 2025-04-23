import norse
from utils import utils
import utils.YOLOModel as ym
from utils.Events import Events
import os
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import torch
import torchvision
import utils.YOLOTrain as yt
import utils.YOLOModel as ym
from tqdm import tqdm



def preprocess(prediction, conf_thre=0.5, nms_thre=0.7, bbox_per_cell=3, rd = False):
    grid_size = prediction.shape[0]  # Assume square grid
    frame_width, frame_height = 640, 480

    # Prepare the detections list
    detections = []

    # Iterate over all grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(bbox_per_cell):
                # Extract prediction for one bounding box
                bbox_start = k * 5  # Start index for bbox data (x, y, w, h, confidence)
                bbox = prediction[i, j, bbox_start:bbox_start + 5]
                if torch.sum(bbox[:4] == torch.tensor([0,0,0,0],  device=device)) == 4:
                    continue
                obj_conf = bbox[4]  # Objectness confidence

                # Extract class probabilities
                class_probs = prediction[i, j, bbox_per_cell * 5:]
                class_conf, class_pred = torch.max(class_probs, 0)  # Max class confidence and index
                # Apply confidence threshold
                if obj_conf * class_conf >= conf_thre:
                    # Get bounding box coordinates
                    top_x = (bbox[0] + j) / grid_size * frame_width  # Top-left x in image space
                    top_y = (bbox[1] + i) / grid_size * frame_height  # Top-left y in image space
                    width = bbox[2] * frame_width  # Width in image space
                    height = bbox[3] * frame_height  # Height in image space

                    # Compute (x1, y1, x2, y2)
                    x1 = top_x - width / 2
                    y1 = top_y - height / 2
                    x2 = x1 + width
                    y2 = y1 + height
                    # if rd:
                    #     x1, y1, x2, y2 = 640*x1, 480*y1, 640*x2, 480*y2
                    # Append detection: (x1, y1, x2, y2, confidence, class_id)
                    detections.append([x1, y1, x2, y2, obj_conf * class_conf, class_pred.item()])

    # Apply Non-Maximum Suppression (NMS) if detections are not empty
    if len(detections) > 0:
        detections = torch.tensor(detections)
        nms_indices = torchvision.ops.nms(
            detections[:, :4],  # Coordinates (x1, y1, x2, y2)
            detections[:, 4],   # Confidence scores
            nms_thre            # IoU threshold
        )
        detections = detections[nms_indices]

    return detections





trainset = [
            # 'interlaken_00_a',
            # 'interlaken_00_b',
            # 'interlaken_00_c',
            # 'interlaken_00_d',
            # 'interlaken_00_e',
            # 'interlaken_00_f',
            # 'thun_00_a',
            # 'thun_01_a',
            # 'thun_01_b',
            # 'zurich_city_03_a',
            # 'zurich_city_04_a',
            # 'zurich_city_04_b',
            # 'zurich_city_04_c',
            # 'zurich_city_04_d',
            # 'zurich_city_04_e',
            # 'zurich_city_05_a',
            # 'zurich_city_05_b',
            # 'zurich_city_06_a',
            # 'zurich_city_07_a',
            # 'zurich_city_08_a',
            # 'zurich_city_11_a',
            # 'zurich_city_11_b',
            # 'zurich_city_11_c',
            # 'zurich_city_13_a',
            # 'zurich_city_13_b',
            # 'zurich_city_15_a',
            'interlaken_01_a',
            'zurich_city_09_a',
            'zurich_city_09_b',
            'zurich_city_09_c',
            'zurich_city_09_d',
            'zurich_city_09_e',
            'zurich_city_10_a',
            'zurich_city_10_b',
            'zurich_city_12_a',
            'zurich_city_14_a',
            'zurich_city_14_b',
            'zurich_city_14_c',
            'zurich_city_00_a',
            'zurich_city_00_b',
            'zurich_city_01_a',
            'zurich_city_01_b',
            'zurich_city_01_c',
            'zurich_city_01_d',
            'zurich_city_01_e',
            'zurich_city_01_f',
            'zurich_city_02_a',
            'zurich_city_02_b',
            'zurich_city_02_c',
            'zurich_city_02_d',
            'zurich_city_02_e',
            ]


def get_video(frames, outputs, output_video):
    height, width = frames[0].shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(output_video, fourcc, 20, (width, height))
    for idx, out in enumerate(outputs):
        frame = utils.postprocess_YOLO(out, frames[idx], 9)
        plt.figure(figsize=(10, 10))
        frame = (frame/255)
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


device='cuda:1'
batch_size=1
grid_size=12
bin_time = 10
seq_length = 50
trainset = yt.HDF5Dataset_YOLO(trainset, bin_time, seq_length)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=16)

model_name = 'yolov1_model_v2_on_Glare_clean.pt'

model = ym.YOLOBase(grid_size=grid_size, num_classes=9, bbox_per_cell=3)
model = norse.torch.Lift(model)
model.load_state_dict(torch.load(model_name, weights_only=True))
model = model.to(device)
model.eval()

losses = []
outs = []
old_filename = ''
frames = []
outputs = []
for epoch in range(1):
    for data, target, flag, filename in tqdm(trainloader):
        if old_filename == '':
            old_filename = filename 
        data, target = data.to(device), target.to(device)
        if len(target) != batch_size:
            continue
        target = yt.process_labels(target, batch_size, grid_size, 9, bbox_per_cell=3)
        # plt.imshow((20*data[-1][0]).clip(0, 255).to('cpu'))
        # plt.savefig('data_plt.png')
        data = data.permute(1, 0, 4, 2, 3).float()
        output = model(data)
        # out = preprocess(output[-1][0])
        if filename != old_filename:
            output_video = f'../videos/network_res/{old_filename}.mp4'
            get_video(frames, outputs, output_video)
            old_filename = filename
            frames = []
            outputs = []
        outputs.append(output[-1][0].cpu().detach())
        frames.append(data[-1][0].cpu().detach())
        # outs.append(out)
        # print(ious)
        # print(outs)
        # print(f'ious_{model_name}')
        # torch.save(outs, f'output_{model_name}')