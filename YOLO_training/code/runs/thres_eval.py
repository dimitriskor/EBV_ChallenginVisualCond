'''
Create the dataset splitting into train/test (val?)
    - Option for random / dawn (manual) / night (manual) / grale (auto)

get all the names of the files
copy all the content in the folder TS_yolo/data/DSEC/option/TS_method/train(or val)/images appending the name of the recording in the beginning
For each recording, load from the loader (copy code?) the labels and add them as txt in the labels folder

'''

import os
import argparse
import shutil
import pickle
import torch
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data import remap, fetch_data_to_write
from tqdm import tqdm
from src.utils.utils import optimal_distance
import cv2
from ultralytics import YOLO


args = argparse.ArgumentParser()
args.add_argument('--split', type=str, help='Strategy to split train/val dataset. Options are random / dawn (manual) / night (manual) / grale (auto)')
args.add_argument('--TS', type=str, help='Method of time surface. Options are bins / EROS / GEROS / ConvEROS / rgb')
args.add_argument('--classes', type=int, help='Number of classes to train on')
args.add_argument('--dataset', type=str, help='Generate dummy dataset type (train, hold, val). Val get -15 and +5 frames. Hold is zurich 11 and 6.')
args = args.parse_args()
relaxation = 0.02

recording_names = os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/')
# images_prefix = f'../../../representations/{args.TS}/'
images_prefix = f'../../../../E2F/results_short_small_inf/'
rgb_prefix = '/mnt/raid0a/Dimitris/DSEC/images_recordings/'

model_name = 'e2f_final/x_True_True3/weights/best.pt'
conf_thr = 0.5


def run_images(img_events, img_rgb):
    results_events = model_events.predict(img_events)
    results_rgb = model_events.predict(img_rgb)

    num_of_pred_obj = 0
    for box in results_events[0].boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        # label = model.names[class_id]

        if confidence > conf_thr:
            num_of_pred_obj += 1
            # x1, y1, x2, y2 = map(int, box.xyxy[0])
            # cv2.rectangle(event_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(event_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    pred_obj_in_events.append(num_of_pred_obj)

    num_of_pred_obj_rgb = 0
    for box in results_rgb[0].boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        # label = model.names[class_id]

        if confidence > conf_thr:
            num_of_pred_obj_rgb += 1
    pred_obj_in_rgb.append(num_of_pred_obj_rgb)




pred_obj = []
gt_obj = []
flagged_sequences = []

for rec_name in recording_names:
    if 'zurich_city_03' in rec_name or 'zurich_city_09' in rec_name or 'zurich_city_10' in rec_name or 'zurich_city_12' in rec_name: # or 'zurich_city_01' in rec_name or 'zurich_city_02' in rec_name:
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
    
    for idx, image in enumerate(images):
        if '.png' not in image:
            continue
        if idx < 5:
            continue

        low = max(0, idx-110)
        high = min(idx+70, len(images))
        is_flagged = distance[idx] > 0.035
        flags.append(is_flagged)
        if args.dataset == 'val':
            if (np.max(distance[low:high] > 0.035) and ('zurich_city_01' not in rec_name or 'zurich_city_02' not in rec_name)):
                im1 = cv2.imread(images_prefix+rec_name+'/'+image)
                img = im1
                img_rgb = cv2.imread(rgb_prefix+f'{rec_name}/images/left/rectified/0{image}')
                run_images(img, img_rgb)
            
        if args.dataset == 'hold':
            if (np.max(distance[low:high] < 0.035) and ('zurich_city_06' in rec_name or 'zurich_city_11_a' in rec_name)):
                im1 = cv2.imread(images_prefix+rec_name+'/'+image)
                img = im1
                img_rgb = cv2.imread(rgb_prefix+f'{rec_name}/images/left/rectified/0{image}')
                run_images(img, img_rgb)

        if args.dataset == 'train':
            if (np.max(distance[low:high] < 0.035) and not ('zurich_city_06' in rec_name or 'zurich_city_11_a' in rec_name)):
                im1 = cv2.imread(images_prefix+rec_name+'/'+image)
                img = im1
                img_rgb = cv2.imread(rgb_prefix+f'{rec_name}/images/left/rectified/0{image}')
                run_images(img, img_rgb)
    pred_obj.append(pred_obj_in_events)
    gt_obj.append(pred_obj_in_rgb)
    flagged_sequences.append(flags)