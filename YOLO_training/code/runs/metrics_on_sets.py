import os
import argparse
import shutil
import pickle
import torch
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './..')))
from src.utils.data import remap, fetch_data_to_write
from tqdm import tqdm
from src.utils.utils import optimal_distance
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

target_labels = {'car', 'bus', 'truck'}
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def run_images(img_events, img_rgb, model_events, model_rgb, rec_name):
    results_events = model_events.track(img_events, persist=True, verbose=False)
    results_rgb = model_rgb.track(img_rgb, persist=True, verbose=False)
    event_boxes = []
    for box in results_events[0].boxes:
        confidence = float(box.conf[0])
        if confidence > conf_thr:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            event_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(img_events, (x1, y1), (x2, y2), (0, 255, 0), 2)

    rgb_boxes = []
    filtered_boxes_rgb = []
    for box in results_rgb[0].boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model_rgb.names[class_id]

        if confidence > conf_thr and label in target_labels:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if np.abs(x1-x2)/1440 < 0.025 or np.abs(y1-y2)/1080 < 0.025:
                continue
            rgb_boxes.append([x1/1440, y1/1080, x2/1440, y2/1080])
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box_coord = box.xyxy.cpu()
            filtered_boxes_rgb.append(torch.tensor([box_coord[0, 0], box_coord[0, 1], box_coord[0, 2], box_coord[0, 3], confidence, class_id]))
    
    remapped_labels = remap(filtered_boxes_rgb, rec_name, '/mnt/raid0a/Dimitris/DSEC/event_recordings/')[0]


    # Match predictions using IoU ≥ 0.5
    matched_events = set()
    matched_rgb = set()
    for i, ebox in enumerate(event_boxes):
        for j, rbox in enumerate(remapped_labels):
            if compute_iou(ebox, rbox) >= 0.1:
                matched_events.add(i)
                matched_rgb.add(j)
                break  # Only one match per event box

    TP = len(matched_events)
    FP = len(event_boxes) - TP
    FN = len(rgb_boxes) - TP

    true_positives.append(TP)
    false_positives.append(FP)
    false_negatives.append(FN)





recording_names = os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/')
# images_prefix = f'../../representations/EROS/'
images_prefix = f'../../representations/bins_40_clip_8/'
# images_prefix = f'../../../E2F/final_res/'
rgb_prefix = '/mnt/raid0a/Dimitris/DSEC/images_recordings/'

# model_name = 'EROS/x_True_False/weights/best.pt'
model_name = 'glare_bins_40_clip_8_TS_1/train/weights/best.pt'
# model_name = 'e2f_35/x_True_True/weights/epoch20.pt'
conf_thr = 0.35
dataset_type = 'hold'
pred_obj = []
gt_obj = []
flagged_sequences = []

true_positives = []
false_positives = []
false_negatives = []

for rec_name in recording_names:
    if dataset_type == 'val':
        if 'zurich_city_04_f' not in rec_name and 'interlaken_00_b' not in rec_name and 'interlaken_01_a' not in rec_name:
            continue
        if 'zurich_city_04_f' == rec_name:
            gl_seq = [(641+50, 641+110), (641+150, 641+210)]
        if 'interlaken_00_b' == rec_name:
            gl_seq = [(665+40, 665+140), (665+155, 665+315)]
        if 'interlaken_01_a' == rec_name:
            gl_seq = [(314+75, 314+125)]#, (1968+0, 1968+125)]
    else:
        # if 'zurich_city_06_a' == rec_name:
        #     gl_seq = [(0, len(os.listdir(images_prefix+rec_name))-1)]
        if 'zurich_city_11_a' == rec_name:
            gl_seq = [(0, len(os.listdir(images_prefix+rec_name))-1)]
        # elif 'zurich_city_12_a' == rec_name:
        #     gl_seq = [(0, len(os.listdir(images_prefix+rec_name))-1)]
        elif 'zurich_city_03_a' == rec_name:
            gl_seq = [(0, len(os.listdir(images_prefix+rec_name))-1)]
        elif 'zurich_city_10_a' == rec_name:
            gl_seq = [(0, len(os.listdir(images_prefix+rec_name))-1)]
        else: 
            continue

    print(rec_name)
    images = os.listdir(images_prefix+rec_name)
    images.sort()
    optimal_distance_val = optimal_distance('DSEC')[0]
    metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  rec_name)
    metric = metric['partial_contrast']
    distance = np.sum(np.square((metric-optimal_distance_val)), axis=1)
    is_in_test = []
    model_rgb = YOLO('yolo11x')
    model_events = YOLO(model_name)
    pred_obj_in_events = []
    pred_obj_in_rgb = []
    flags = []
    prev = 0
    
    for idx, image in enumerate(images):
        for seq in gl_seq:
            if idx < seq[0] or idx > seq[1]:
                continue
            if '.png' not in image:
                continue
            if idx < 5:
                continue

            im1 = cv2.imread(images_prefix+rec_name+'/'+image)
            img = im1
            img_rgb = cv2.imread(rgb_prefix+f'{rec_name}/images/left/rectified/0{image}')
            # img_rgb = cv2.imread(rgb_prefix+f'{rec_name}/images/left/rectified/0{image[6:]}')
            run_images(img, img_rgb, model_events, model_rgb, rec_name)

total_TP = sum(true_positives)
total_FP = sum(false_positives)
total_FN = sum(false_negatives)

print(total_TP, total_FP, total_FN)
print("Precision:", total_TP/(total_TP+total_FP))
print("Recall:", total_TP/(total_TP+total_FN))