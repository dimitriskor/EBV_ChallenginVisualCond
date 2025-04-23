from ultralytics import YOLO
import cv2
import os
import torch
from ultralytics.engine.results import Boxes
import pickle
from utils.Frames import Frames 



def run(videos, model_type, train_test):
    model = YOLO(f"yolo{model_type}.pt")

    specific_labels = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
    # Set confidence threshold (e.g., 0.5)
    conf_threshold = 0.5


    for v in videos:
        path = f'../../data/DSEC/{train_test}/{v}'
        print(v)
        stats = {'video': v,
                'frames': 0,
                'person' : 0,
                'car' : 0,
                'bicycle' : 0,
                'motorcycle' : 0,
                'bus' : 0,
                'truck' : 0,
                'traffic light' : 0,
                'stop sign' : 0,
                }    
        new_class_id = [
                'person',
                'car',
                'bicycle',
                'motorcycle',
                'bus',
                'truck',
                'traffic light',
                'stop sign']

        detections_list = []
        height, width = 1080, 1440
        frames_file = Frames(path)
        filename = path.split('/')[-1]

        for frame in frames_file.frames_as_array:


            # Run YOLO11 tracking on the frame, persisting tracks between frames
            try:
                frame_detection = []
                filtered_boxes = []
                # Perform tracking with YOLOv8
                print(frame.shape)
                results = model.track(frame, persist=True)

                # Extract detections from results
                detections = results[0].boxes  # Get detected boxes from results
                stats['frames'] += 1
                for det in detections:
                    # Extract the confidence score and class label name
                    conf = det.conf.item()  # Get confidence score
                    class_name = results[0].names[int(det.cls.item())]  # Get the class label name

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

                annotated_frame = results[0].plot()
                print(annotated_frame.shape)

            except Exception as e:
                pass
                print('pass', e)
            detections_list.append(filtered_boxes)
        with open(f'../../data/DSEC/object_detection_yolo/{filename}.pkl', 'wb') as f:
            pickle.dump(detections_list, f)

all_videos = list(os.listdir('../../data/DSEC/test_images'))
videos = []
for v in all_videos:
    # if 'no_scatter' in v:
    videos.append(v)


# model_type = '11n'

# run(videos, '11n', 'test_images')
run(videos, '11m', 'test_images')
# run(videos, '11x', 'test_images')
# run(videos, 'v3', 'test_images')


all_videos = list(os.listdir('../../data/DSEC/train_images'))
videos = []
for v in all_videos:
    # if 'no_scatter' in v:
    videos.append(v)


# run(videos, '11n', 'train_images')
run(videos, '11m', 'train_images')
# run(videos, '11x', 'train_images')
# run(videos, 'v3', 'train_images')