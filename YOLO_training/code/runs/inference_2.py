from ultralytics import YOLO
import cv2
import os
import torch
from ultralytics.engine.results import Boxes
import pickle



specific_labels = ['car']
# Set confidence threshold (e.g., 0.5)
conf_threshold = 0.5


stats = {'video': '',
        'frames': 0,
        'car' : 0,
        }    
new_class_id = [
        'car'
        ]




detections_list = []
frames_folder = '../../TS_yolo/dataset_1/glare/bins_40_clip_8/val/images/'
labels_folder = '../../TS_yolo/dataset_1/glare/bins_40_clip_8/val/labels/'
filenames = os.listdir(frames_folder)
labelnames = os.listdir(labels_folder)

filenames.sort()
labelnames.sort()
batches_images = []
batches_labels = []
current_batch_images = [filenames[0]]
current_batch_labels = [labelnames[0]]
for i in range(1, len(filenames)):
    if int(filenames[i].split('.')[0].split('_')[-1]) == int(filenames[i-1].split('.')[0].split('_')[-1]) + 1:  # Check if consecutive
        current_batch_images.append(filenames[i])
        current_batch_labels.append(labelnames[i])
    else:
        batches_images.append(current_batch_images)
        batches_labels.append(current_batch_labels)
        current_batch_images = [filenames[i]]
        current_batch_labels = [labelnames[i]]

# Add the last batch
batches_images.append(current_batch_images)
batches_labels.append(current_batch_labels)

alvis_models_pref = 'e2f/'
model_dir = os.listdir(alvis_models_pref)


for model_path in model_dir:
    model_types = ['/weights/best.pt', '/weights/last.pt']
    for m_type in model_types:
        model_name = alvis_models_pref + model_path + m_type
        print(model_name)
        gt_obj = []
        pred_obj = []
        try:
            for i in range(len(batches_images)):

                model = YOLO(model_name)

                batch_gt_obj = []
                batch_pred_obj = []
                for j in range(len(batches_images[i])):
                    frame_name = batches_images[i][j]
                    label_name = batches_labels[i][j]
                    # Run YOLO11 tracking on the frame, persisting tracks between frames
                    # try:
                    frame_detection = []
                    filtered_boxes = []
                    frame = cv2.imread('../../TS_yolo/dataset_1/glare/e2f_dummy_val/val/images/'+frame_name)
                    with open('../../TS_yolo/dataset_1/glare/e2f_dummy_val/val/labels/'+label_name, 'r') as f:
                        labels = f.read()
                        num_of_gt_objects = len(labels.split('\n')) - 1

                    # Perform tracking with YOLOv8
                    results = model.track(frame, persist=True, verbose=False)

                    # Extract detections from results
                    detections = results[0].boxes  # Get detected boxes from results
                    
                    num_of_pred_objects = len(detections)

                    batch_gt_obj.append(num_of_gt_objects)
                    batch_pred_obj.append(num_of_pred_objects)
                
                gt_obj.append(batch_gt_obj)
                pred_obj.append(batch_pred_obj)
            with open(model_name[:-3]+'_gt_objects.pkl', 'wb') as f:
                pickle.dump(gt_obj, f)
            with open(model_name[:-3]+'_pred_objects.pkl', 'wb') as f:
                pickle.dump(pred_obj, f)
        except FileNotFoundError as e:
            print(e)