import torch
from ultralytics.engine.results import Boxes
import pickle
import argparse
import os
from ultralytics import YOLO
import cv2

dt_type = 'val'
event_videos = os.listdir('e2f_gray_trained/x_True_True/weights/')
event_videos = [evv for evv in event_videos if ('_event.mp4' in evv) and ('annotated' not in evv) and (dt_type in evv)]

rgb_videos = os.listdir('e2f_gray_trained/x_True_True/weights/')
rgb_videos = [rgbv for rgbv in rgb_videos if '_rgb.mp4' in rgbv and ('annotated' not in rgbv) and (dt_type in rgbv)]

prefix = 'e2f_gray_trained/x_True_True/weights/'
target_labels = {'car', 'bus', 'truck'}
 

    
pred_objects = []
for evv in event_videos:
    ebv_yolo = YOLO(f"e2f_gray_trained/x_True_True/weights/epoch22.pt")
    results = ebv_yolo.track(source=prefix+evv, persist=True, show=True, verbose=False)
    cap = cv2.VideoCapture(prefix+evv)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(prefix+'annotated_'+evv, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    pred_obj_in_frames = []
    for result in results:
        frame = result.orig_img.copy()
        num_of_pred_obj = 0
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = ebv_yolo.names[class_id]

            if confidence > 0.5:
                num_of_pred_obj += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pred_obj_in_frames.append(num_of_pred_obj)
        out.write(frame)


    pred_objects.append(pred_obj_in_frames)
    cap.release()
    out.release()

with open(f'e2f_gray_trained/x_True_True/weights/pred_objects_{dt_type}_extended.pkl', 'wb') as f:
    pickle.dump(pred_objects, f)



gt_objects = []
for rgbv in rgb_videos:
    rgb_yolo = YOLO('yolo11x')
    results = rgb_yolo.track(source=prefix+rgbv, persist=True, show=True, verbose=False)
    cap = cv2.VideoCapture(prefix+rgbv)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(prefix+'annotated_'+rgbv, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    gt_obj_in_frames = []
    for result in results:
        frame = result.orig_img.copy()
        num_of_gt_obj = 0
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = rgb_yolo.names[class_id]

            # if label in target_labels:
            if confidence > 0.5 and label in target_labels:
                num_of_gt_obj += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        gt_obj_in_frames.append(num_of_gt_obj)
        out.write(frame)

    gt_objects.append(gt_obj_in_frames)
    cap.release()
    out.release()

with open(f'e2f_gray_trained/x_True_True/weights/gt_objects_{dt_type}_extended.pkl', 'wb') as f:
    pickle.dump(gt_objects, f)


