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
from utils.visualize import fetch_data_to_write, get_slices
from utils.utils import optimal_distance, postprocess, get_annotated_frame
import torchvision


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


def run_inference(model, data, folder, num_classes=9):
    '''
        Run model on inference and create video with the output.
    '''
    print(type(data))
    output_dir = f'../statistics/network_res/{tr_te}/{folder}.pt'
    extra_channel = data[:, 0, :, :].unsqueeze(1)
    print(data.shape, 'before permute')
    data = torch.cat((data, extra_channel), dim=1).unsqueeze(1).float()
    print(data.shape, 'after permute')
    output = model(data)
    outs = []
    for idx, out in enumerate(output[10:]):
        out = preprocess(output[idx][0])
        outs.append(out)
    torch.save(outs, output_dir)
    

def exclude_elements(main_list, remove_sublists):
    flat_main_list = main_list[0]

    # Create a set of indices to exclude
    exclude_indices = set()
    for sublist in remove_sublists:
        start = flat_main_list.index(sublist[0])
        for i in range(len(sublist)):
            exclude_indices.add(start + i)

    # Construct the resulting list
    result = []
    current_sublist = []

    for idx, value in enumerate(flat_main_list):
        if idx not in exclude_indices:
            current_sublist.append(value)
        elif current_sublist:
            result.append(current_sublist)
            current_sublist = []

    if current_sublist:  # Append any remaining elements
        result.append(current_sublist)
    return result


device = 'cpu'
grid_size = 12

model = ym.YOLOBase(grid_size=grid_size, num_classes=9, bbox_per_cell=3)
model = norse.torch.Lift(model)
model.load_state_dict(torch.load('yolov1_model_v2_on_Glare_clean.pt', weights_only=True))
model = model.to(device)
model.eval()
folder_names = list(os.listdir('/mnt/raid0a/Dimitris/DSEC/train_events/'))
prefix = '/mnt/raid0a/Dimitris/DSEC/train_events/'
tr_te = 'train'







opt_distance = optimal_distance('DSEC')[0]

for folder in folder_names:
    # if '10_a' in folder or '04_f' in folder or '10_b' in folder or '09_b' in folder or '09_c' in folder:
    #     continue
    # if '00_b' not in folder:
    #     continue
    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=10, height=480, width=640)
    metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  folder)
    metric = metric['partial_contrast']
    distance = np.sum(np.square((metric-opt_distance)), axis=1)
    slices = get_slices(distance)
    all_slices = [list(range(0, data.ms_to_idx.shape[0] // 50))]
    slices = exclude_elements(all_slices, slices)
    if slices == []:
        continue
    for s_id, slice in enumerate(slices):
        objects = [0, 0, 0, 0] # frames, class 0, ... 
        count = 0
        while len(slice) > 500:
            slice, slice_left = slice[:500], slice[500:] 
            events = torch.tensor(data.events_to_frames_DSEC(slice))
            events = events.to(device)
            run_inference(model, events, folder+str(count)+'_'+str(s_id))
            slice = slice_left
            count += 1
        events = torch.tensor(data.events_to_frames_DSEC(slice))
        run_inference(model, events, folder+str(count)+'_'+str(s_id))
    del data






folder_names = list(os.listdir('/mnt/raid0a/Dimitris/DSEC/test_events/'))
prefix = '/mnt/raid0a/Dimitris/DSEC/test_events/'
# tr_te = 'test'

for folder in folder_names:
    # if '10_a' in folder or '04_f' in folder or '10_b' in folder or '09_b' in folder or '09_c' in folder:
    #     continue
    # if '00_b' not in folder:
    #     continue
    print(folder)
    filename = prefix+folder+'/events/left/events.h5'
    data = Events(filename, stack_events=5, height=480, width=640)
    metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  folder)
    metric = metric['partial_contrast']
    distance = np.sum(np.square((metric-opt_distance)), axis=1)
    slices = get_slices(distance)
    all_slices = [list(range(0, data.ms_to_idx.shape[0] // 50))]
    slices = exclude_elements(all_slices, slices)

    if slices == []:
        continue
    for s_id, slice in enumerate(slices):
        objects = [0, 0, 0, 0] # frames, class 0, ... 
        count = 0
        while len(slice) > 500:
            slice, slice_left = slice[:500], slice[500:] 
            events = torch.tensor(data.events_to_frames_DSEC(slice))
            events = events.to(device)
            run_inference(model, events, folder+str(count)+'_'+str(s_id))
            slice = slice_left
            count += 1
        events = torch.tensor(data.events_to_frames_DSEC(slice))
        run_inference(model, events, folder+str(count)+'_'+str(s_id))
    del data
