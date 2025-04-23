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
        'car',
        'truck',
        'bus',
        ]




detections_list = []
# frames_folder = '../../../E2F/results/'
# labels_folder = '../../../E2F/results/'
dt_type = 'val'
frames_folder = f'../../TS_yolo/dataset_1/glare/e2f_final{dt_type}/val/images/'
labels_folder = f'../../TS_yolo/dataset_1/glare/e2f_final{dt_type}/val/labels/'
filenames = os.listdir(frames_folder)
labelnames = os.listdir(labels_folder)

filenames.sort()
labelnames.sort()
for f in filenames:
    if '0mps' in f:
        filenames.remove(f)
        print(f)
        
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

for i in range(len(batches_images)):
    name = '_'.join(batches_images[i][0].split('_')[:-1])
    first = batches_images[i][0]
    first = int(first.split('.')[0].split('_')[-1])
    last = batches_images[i][-1]
    last = int(last.split('.')[0].split('_')[-1])
    for j in range(60):
        batches_images[i].insert(j, name+'_'+f'{first-60+j:05}'+'.png')
        batches_images[i].append(name+'_'+f'{last+1+j:05}'+'.png')



alvis_models_pref = 'e2f_final/'
model_dir = os.listdir(alvis_models_pref)

target_labels = {'car', 'bus', 'truck'}

pred_objects = []
gt_objects = []

for model_path in model_dir:
    if 'x_True_True3' not in model_path:
        continue
    model_types = ['/weights/best.pt']
    for m_type in model_types:
        model_name = alvis_models_pref + model_path + m_type
        print(model_name)
        try:
            for i in range(len(batches_images)):
                # if i != 17:
                #     continue
                model = YOLO(model_name)
                model_rgb = YOLO('yolo11x')

                
                

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_event = cv2.VideoWriter(model_name[:-3]+f'_batch_{i}_{dt_type}_event_short.mp4', fourcc, 20, (640, 480), isColor=True)
                out_rgb = cv2.VideoWriter(model_name[:-3]+f'_batch_{i}_{dt_type}_rgb_short.mp4', fourcc, 20, (1440, 1080), isColor=True)
                gt_obj_in_frames = []
                pred_obj_in_frames = []
                for j in range(len(batches_images[i])):
                    frame_name = batches_images[i][j]
                    recording = '_'.join(frame_name.split('_')[:-1])
                    frame_id = frame_name.split('_')[-1]
                    event_frame = cv2.imread(f'../../../E2F/results_short_small_inf/{recording}/frame_{frame_id}')
                    rgb_frame = cv2.imread(f'/mnt/raid0a/Dimitris/DSEC/images_recordings/{recording}/images/left/rectified/0{frame_id}')

                    results_events = model.track(event_frame, persist=True, verbose=False)
                    results_rgb = model_rgb.track(rgb_frame, persist=True, verbose=False)
                    num_of_pred_obj = 0
                    for box in results_events[0].boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = model.names[class_id]

                        if confidence > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            num_of_pred_obj += 1
                            cv2.rectangle(event_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(event_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    pred_obj_in_frames.append(num_of_pred_obj)
                    out_event.write(event_frame)
                    num_of_gt_obj = 0
                    for box in results_rgb[0].boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = model_rgb.names[class_id]

                        # if label in target_labels:
                        if confidence > 0.5 and label in target_labels:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            if (x2-x1)/1440 < 0.01 or (y2-y1)/1080 < 0.01:
                                continue
                            num_of_gt_obj += 1
                            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(rgb_frame, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    gt_obj_in_frames.append(num_of_gt_obj)
                    out_rgb.write(rgb_frame)
                pred_objects.append(pred_obj_in_frames)
                gt_objects.append(gt_obj_in_frames)

                
        except FileNotFoundError as e:
            print(e)

    with open(f'e2f_gray_trained/m_True_True4/weights/pred_objects_{dt_type}_extended.pkl', 'wb') as f:
        pickle.dump(pred_objects, f)
    with open(f'e2f_gray_trained/m_True_True4/weights/gt_objects_{dt_type}_extended.pkl', 'wb') as f:
        pickle.dump(gt_objects, f)