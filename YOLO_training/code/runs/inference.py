from ultralytics import YOLO
import cv2
import os
import torch
from ultralytics.engine.results import Boxes
import pickle



model = YOLO(f"e2f/val_x_True_True2/weights/epoch40.pt")
# model = YOLO('yolo11x')

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
# frames_folder = '/mnt/raid0a/Dimitris/DSEC/images_recordings/zurich_city_04_f/images/left/rectified/'
frames_folder = '../../../E2F/results/zurich_city_04_f/'
filenames = os.listdir(frames_folder)
filenames.sort()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test.mp4', fourcc, 20, (640, 480), isColor=True)
for frame_name in filenames:

    print(frame_name)
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    try:
        frame_detection = []
        filtered_boxes = []
        frame = cv2.imread(frames_folder+frame_name)
        # Perform tracking with YOLOv8
        results = model(frame)

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
            if conf >= conf_threshold and class_name in specific_labels:
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



        # results = model.track(frame, persist=True)
        print(filtered_boxes)
        # Visualize the results on the frame
        # frame *= 20
        annotated_frame = results[0].plot()
        out.write(annotated_frame)


    except Exception as e:
        pass
        print('pass', e)
    detections_list.append(filtered_boxes)

out.release()