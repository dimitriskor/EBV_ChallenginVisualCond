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
        'bus',
        'truck'
        ]




detections_list = []
# frames_folder = '../../../E2F/results/'
# labels_folder = '../../../E2F/results/'
dt_type = 'val'
frames_folder = f'../../TS_yolo/dataset_1/glare/e2f_final/val/images/'
# frames_folder = f'../../TS_yolo/dataset_1/glare/e2f_temp_{dt_type}/val/images/'
labels_folder = f'../../TS_yolo/dataset_1/glare/e2f_final/val/labels/'
# labels_folder = f'../../TS_yolo/dataset_1/glare/e2f_temp_{dt_type}/val/labels/'
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

alvis_models_pref = 'e2f_final/'
# alvis_models_pref = 'e2f_temp/'
model_dir = os.listdir(alvis_models_pref)


for model_path in model_dir:
    if 'm_True_True3' not in model_path:
        continue
    model_types = ['/weights/last.pt']
    for m_type in model_types:
        model_name = alvis_models_pref + model_path + m_type
        print(model_name)
        # if 'val_x_True_True2' not in model_name:
        #     continue
        gt_obj = []
        pred_obj = []
        try:
            for i in range(len(batches_images)):
                model = YOLO(model_name)

                batch_gt_obj = []
                batch_pred_obj = []

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(model_name[:-3]+f'_batch_{i}_{dt_type}.mp4', fourcc, 20, (640, 480), isColor=True)

                for j in range(len(batches_images[i])):
                    frame_name = batches_images[i][j]
                    label_name = batches_labels[i][j]
                    # Run YOLO11 tracking on the frame, persisting tracks between frames
                    # try:
                    frame_detection = []
                    filtered_boxes = []
                    frame = cv2.imread(f'../../TS_yolo/dataset_1/glare/e2f_final/val/images/'+frame_name)
                    # frame = cv2.imread(f'../../TS_yolo/dataset_1/glare/e2f_temp_{dt_type}/val/images/'+frame_name)
                    with open(f'../../TS_yolo/dataset_1/glare/e2f_final/val/labels/'+label_name, 'r') as f:
                    # with open(f'../../TS_yolo/dataset_1/glare/e2f_temp_{dt_type}/val/labels/'+label_name, 'r') as f:
                        labels = f.read()
                        num_of_gt_objects = len(labels.split('\n')) - 1

                    # Perform tracking with YOLOv8
                    results = model.predict(frame,verbose=False)

                    # Extract detections from results
                    detections = results[0].boxes  # Get detected boxes from results
                    stats['frames'] += 1
                    for det in detections:
                        # Extract the confidence score and class label name
                        conf = det.conf.item()  # Get confidence score
                        class_name = results[0].names[int(det.cls.item())]  # Get the class label name
                        track_id = det.id.item() if det.id is not None else None
                        print(f"Track ID: {track_id}, Class: {class_name}, Confidence: {conf}")
                        # Apply confidence threshold and filter by specific labels
                        if conf >= conf_threshold:
                            # Check the structure of det.xyxy
                            box = det.xyxy.cpu()  # Ensure it's on the CPU
                            
                            stats[class_name] += 1
                            # Print the shape of the box for debugging
                            print(f"Box shape: {box.shape}")  # Debugging line to check the shape
                            
                            # Append the box only if it has the expected shape
                            if box.ndim == 2 and box.shape[1] == 4:  # Check if it's a 2D tensor with 4 columns
                                frame_detection.append([box[0, 0], box[0, 1], box[0, 2], box[0, 3], conf, class_name])
                                filtered_boxes.append(torch.tensor([box[0, 0], box[0, 1], box[0, 2], box[0, 3], conf, new_class_id.index(class_name)]))
                            else:
                                print("Unexpected box shape or structure.")  # Error message for unexpected shape

                    # If there are filtered detections, rebuild the Boxes object
                    if filtered_boxes:
                        # Concatenate all filtered boxes into a single tensor
                        filtered_boxes = torch.stack(filtered_boxes)  # Shape: (N, 6)
                        
                        # Specify the original image shape (height, width)
                        orig_shape = (frame.shape[0], frame.shape[1])  # Assuming frame shape is (H, W, C)

                        # Create a new Boxes object with the required attributes
                        filtered_results = Boxes(filtered_boxes, orig_shape)
                        results[0].boxes = filtered_results
                    else:
                        results[0].boxes = None
                    
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    num_of_pred_objects = len(frame_detection)
                    batch_gt_obj.append(num_of_gt_objects)
                    batch_pred_obj.append(num_of_pred_objects)
                gt_obj.append(batch_gt_obj)
                pred_obj.append(batch_pred_obj)
            with open(model_name[:-3]+f'_gt_objects_{dt_type}.pkl', 'wb') as f:
                pickle.dump(gt_obj, f)
            with open(model_name[:-3]+f'_pred_objects_{dt_type}.pkl', 'wb') as f:
                pickle.dump(pred_obj, f)
        except FileNotFoundError as e:
            print(e)