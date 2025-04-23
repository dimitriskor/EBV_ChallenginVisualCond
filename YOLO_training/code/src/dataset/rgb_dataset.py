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
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.data import remap, fetch_data_to_write
from tqdm import tqdm
from src.utils.utils import optimal_distance

args = argparse.ArgumentParser()
args.add_argument('--split', type=str, help='Strategy to split train/val dataset. Options are random / dawn (manual) / night (manual) / grale (auto)')
args.add_argument('--TS', type=str, help='Method of time surface. Options are bins / EROS / GEROS / ConvEROS / rgb')
args.add_argument('--classes', type=int, help='Number of classes to train on')
args = args.parse_args()
relaxation = 0.01

recording_names = os.listdir('/mnt/raid0a/Dimitris/DSEC/images_recordings/')
# images_prefix = f'../../../representations/{args.TS}/'
images_prefix = f'/mnt/raid0a/Dimitris/DSEC/images_recordings/'
# labels_prefix = '/mnt/raid0a/Dimitris/DSEC/object_detection_yolo/'
labels_prefix = '../../../labels/11x/0.5/'
data_im_prefix = f'../../../TS_yolo/dataset_{args.classes}/{args.split}/{args.TS}/'
data_lbs_prefix = f'../../../TS_yolo/dataset_{args.classes}/{args.split}/{args.TS}/'

if args.classes == 1:
    filter_class = 2

if args.split == 'random':
    recording_names_train = recording_names[:int(len(recording_names)*0.8)]
    recording_names_test = recording_names[int(len(recording_names)*0.8):]


elif args.split == 'glare':
    data_im_prefix_train = data_im_prefix + 'train/images'
    data_lbs_prefix_train = data_lbs_prefix + 'train/labels'
    data_im_prefix_test = data_im_prefix + 'val/images'
    data_lbs_prefix_test = data_lbs_prefix + 'val/labels'
    for rec_name in recording_names:
        if 'zurich_city_03' in rec_name or 'zurich_city_06' in rec_name or 'zurich_city_09' in rec_name or 'zurich_city_10' in rec_name or 'zurich_city_12' in rec_name: # or 'zurich_city_01' in rec_name or 'zurich_city_02' in rec_name:
            continue
        print(rec_name)
        images = os.listdir(images_prefix+rec_name+'/images/left/rectified/')
        images.sort()
        for im in images:
            if '.txt' in im:
                images.remove(im)
        optimal_distance_val = optimal_distance('DSEC')[0]
        metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  rec_name)
        metric = metric['partial_contrast']
        distance = np.sum(np.square((metric-optimal_distance_val)), axis=1)
        is_in_test = []
        for idx, image in enumerate(images):
            if '.png' not in image:
                continue
            # im1 = cv2.imread(images_prefix+rec_name+'/'+image)
            # img = im1
            low = max(0, idx-60)
            high = min(idx+60, len(images))
            if (np.max(distance[low:high] > 0.025) and ('zurich_city_01' not in rec_name or 'zurich_city_02' not in rec_name)) or 'zurich_city_11_a' in rec_name:
                shutil.copyfile(images_prefix+rec_name+'/images/left/rectified/'+image, f'{data_im_prefix_test}/{rec_name}_{image[1:]}')
                is_in_test.append(True)
            elif '.txt' not in rec_name:
                shutil.copyfile(images_prefix+rec_name+'/images/left/rectified/'+image, f'{data_im_prefix_train}/{rec_name}_{image[1:]}')
                is_in_test.append(False)

        target_path = labels_prefix + rec_name + f'.pkl' 
        with open(target_path, 'rb') as f:
            targets = pickle.load(f) 
        for idx, target in enumerate(targets):
            # target = remap(targets[idx], rec_name, '/mnt/raid0a/Dimitris/DSEC/event_recordings/')[0]
            if len(target) != 0:
                target_cx = (target[:, 0] + target[:, 2])/2
                target_w = torch.abs((target[:, 0] - target[:, 2]))
                target_cy = (target[:, 1] + target[:, 3])/2
                target_h = torch.abs(target[:, 1] - target[:, 3])
                target[:, 0] = target_cx/1440
                target[:, 1] = target_cy/1080
                target[:, 2] = target_w/1440
                target[:, 3] = target_h/1080
                target = target[target[:, 2] > relaxation]
                target = target[target[:, 3] > relaxation]
                target = target[:, [5, 0, 1, 2, 3, 4]]
                target = target[:, :5]
                if args.classes != 8:
                    target = torch.cat([target[target[:, 0] == 2], target[target[:, 0] == 5], target[target[:, 0] == 7]], dim=0) 
                    # target = target[target[:, 0] == 2]
                    target[:, 0] = 0
            try:
                if is_in_test[idx]:
                    with open(f'{data_lbs_prefix_test}/{rec_name}_{idx:05}.txt', 'w') as f:
                        line = ''
                        for t in target:
                            line += f'{int(t[0])} {t[1]} {t[2]} {t[3]} {t[4]}\n'
                        f.write(line)
                else:
                    with open(f'{data_lbs_prefix_train}/{rec_name}_{idx:05}.txt', 'w') as f:
                        line = ''
                        for t in target:
                            line += f'{int(t[0])} {t[1]} {t[2]} {t[3]} {t[4]}\n'
                        f.write(line)
            except:
                break

elif args.split == 'night':
    recording_names_train = recording_names[:int(len(recording_names)*0.8)]
    recording_names_test = recording_names[int(len(recording_names)*0.8):]