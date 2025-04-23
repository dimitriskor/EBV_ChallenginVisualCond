from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import h5py
import yaml


def plot_distr_distance(dataset, optimal_distr, vis_type='', image_type='frames'):
    optim_distr_mean, optim_distr_std = optimal_distr
    vis_type = '/' + vis_type 
    path = f'../info_{dataset}/info_data/partial_contrast/{image_type}{vis_type}'
    files = os.listdir(path)
    for file in files:
        # if 'inter' not in file or '00_b' not in file:
        #     continue
        print(file)
        save_name = file.split('.')[0]
        distr = np.load(path+file)
        distance = np.sum(np.square((distr-optim_distr_mean)/(1)), axis=1)
        colors = ['red' if d > 0.025 else 'blue' for d in distance]
        # for i, c in enumerate(colors):
        #     if c == 'red':
        #         print(i)

        plt.figure()
        plt.scatter(np.linspace(0, len(distance)-1, len(distance)), distance, marker='.', s=6, c=colors)
        plt.title(f'Distribution distance from optimal')
        plt.xlabel('Time')
        plt.ylabel('Distance')
        plt.savefig(f'../info_{dataset}/temp_files/{save_name}-d15')
        plt.close()

def optimal_distance(dataset, image_type='frames'):
    if dataset == 'DDD20':
        vis_type = ['/clear', '/glare', '/night']
    elif dataset == 'DSEC':
        vis_type = ['/']
    distr_sum = np.array([], dtype=np.int64).reshape(0, 16)
    # timesteps = 0
    for v in vis_type:
        path = f'../info_{dataset}/info_data/partial_contrast/{image_type}{v}'
        files = os.listdir(path)
        for file in files:
            if 'zurich_city_12_a' in file:
                continue
            distr = np.load(path+file)
            distr_sum = np.concatenate((distr, distr_sum), axis=0)
    return np.mean(distr_sum, axis=0), np.std(distr_sum, axis=0)



def postprocess(prediction, num_classes, conf_thre=0.5, nms_thre=0.7, class_agnostic=False):
    # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2
    prediction[:, :, :4] = box_corner[:, :, :4]
    
    output = [None for _ in range(len(prediction))]
    
    for i, image_pred in enumerate(prediction):
        # If no detections, skip this image
        if not image_pred.size(0):
            continue

        # Get the score and class with the highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        
        # Apply confidence threshold
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        
        if not detections.size(0):
            continue

        # Apply Non-Maximum Suppression (NMS)
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        # Keep only the filtered detections
        detections = detections[nms_out_index]

        # Extract and organize output
        # x1, y1, x2, y2 are in detections[:, :4]
        # Confidence score is detections[:, 4] * detections[:, 5]
        # Class is detections[:, 6]
        box_corners = detections[:, :4]
        confidences = detections[:, 4] * detections[:, 5]
        classes = detections[:, 6]

        # Combine into a single tensor for final output: (x1, y1, x2, y2, confidence, class)
        final_detections = torch.cat((box_corners, confidences.unsqueeze(1), classes.unsqueeze(1)), 1)
        
        if output[i] is None:
            output[i] = final_detections.tolist()
        else:
            output[i] = torch.cat((output[i], final_detections)).tolist()

    return output


from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

def get_annotated_frame(frame: torch.Tensor, results, rgb=True):
    # Convert the frame to a PIL Image for easier drawing
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
    frame = 512*frame/8
    print(frame.shape)
    frame = frame.cpu().byte()  # Ensure the frame is in the byte format
    pil_frame = Image.fromarray(frame.numpy())
    draw = ImageDraw.Draw(pil_frame)

    # Font settings (default font, you can specify a custom TTF if desired)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    if results is not None:
        for res in results:
            print(res)
            [x_min, y_min, x_max, y_max, conf, obj_class] = res
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            obj_class = int(obj_class)

            # Draw the bounding box with color based on the class
            color = (200 * (obj_class == 2), 200 * (obj_class == 1), 200 * (obj_class == 0))
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

            # Prepare the text annotation: confidence and class
            annotation_text = f'Class: {obj_class}, Conf: {conf:.2f}'

            # Calculate the size of the text box using `textbbox`
            if font:
                text_bbox = draw.textbbox((0, 0), annotation_text, font=font)
            else:
                text_bbox = draw.textbbox((0, 0), annotation_text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Define the background rectangle for text
            text_background = (x_min, y_min - text_height - 2, x_min + text_width, y_min)

            # Draw the background rectangle for text
            draw.rectangle(text_background, fill=color)

            # Draw the annotation text on top of the box
            draw.text((x_min, y_min - text_height), annotation_text, fill='white', font=font)

    # Convert the annotated PIL image back to a Torch Tensor
    annotated_frame = torch.tensor(np.array(pil_frame), dtype=torch.uint8)
    # annotated_frame = annotated_frame.permute(1, 2, 0)
    
    return annotated_frame






import torch
import cv2
import matplotlib.pyplot as plt
import torchvision

def postprocess_YOLO(prediction, frame, num_classes, conf_thre=0.5, nms_thre=0.7, bbox_per_cell=3):
    """
    Post-process YOLO predictions, apply NMS, and plot predictions on the image.

    Args:
        prediction (tensor): The YOLO output tensor of shape (grid_size, grid_size, bbox_per_cell * 5 + num_classes).
        frame (numpy array): The image (as a numpy array) where the detections will be plotted.
        num_classes (int): Number of classes in the YOLO model.
        conf_thre (float): Confidence threshold for filtering.
        nms_thre (float): IoU threshold for Non-Maximum Suppression.
        bbox_per_cell (int): Number of bounding boxes per grid cell.

    Returns:
        detections (list): A list of detections with bounding boxes, confidences, and class ids.
    """
    grid_size = prediction.shape[0]  # Assume square grid
    if isinstance(frame, torch.Tensor):
        frame = (20*frame.permute(1, 2, 0)).clip(0, 255).to('cpu').numpy()
        
    frame_height, frame_width = frame.shape[:2]

    # Prepare the detections list
    detections = []

    # Iterate over all grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(bbox_per_cell):
                # Extract prediction for one bounding box
                bbox_start = k * 5  # Start index for bbox data (x, y, w, h, confidence)
                bbox = prediction[i, j, bbox_start:bbox_start + 5]
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

        # Draw bounding boxes on the frame
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det.tolist()

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Define label text and color
            label = f"Class {class_id}: {confidence:.2f}"
            color = (255, 0, 0)  # Red for bounding boxes
            # Draw the bounding box and label
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Plot the image with bounding boxes
    # plt.figure(figsize=(10, 10))
    # plt.imshow(frame/255)
    # plt.savefig('data_plt_boxes.png')
    # plt.axis('off')
    # plt.show()
    return frame
    # return detections.tolist()


import matplotlib.patches as patches


def plot_bounding_boxes(target, frame):
    """
    Plots bounding boxes on a given frame.

    Parameters:
    - target: numpy array or list of shape (object, 5), where each row is (top_x, top_y, width, height, class_label).
    - frame: numpy array of the image to display (H, W, C) or (H, W).
    """
    # Create a matplotlib figure
    fig, ax = plt.subplots(1)
    
    # Show the image
    ax.imshow(frame*20, cmap='gray' if len(frame.shape) == 2 else None)

    # Iterate over each object in the target
    for obj in target:
        top_x, top_y, width, height, _, class_label = obj
        top_x = top_x - width/2
        top_y = top_y - height/2
        print(target)
        # Create a rectangle patch
        rect = patches.Rectangle((top_x*640, top_y*480), width*640, height*480, linewidth=2, edgecolor='red', facecolor='none')
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        # Add class label as text
        ax.text(top_x*640, top_y*480 - 5, str(class_label.item()), color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Display the plot
    plt.axis('off')
    plt.savefig('data_plt_original.png')





def map_pixel_positions(original_pixel_positions, remapping):
    """
    Maps specific pixel positions from the original image to their corresponding
    positions in the new image using the remapping array.

    Args:
        original_pixel_positions: List of [x1, y1, x2, y2, ...] objects, where x1, y1 and x2, y2
                                  are pixel indices in the original image.
        remapping: The remapping array of shape (new_height, new_width, 2).
                   Each entry remapping[new_y, new_x] gives the (orig_x, orig_y) that maps to (new_x, new_y).

    Returns:
        A list of updated pixel positions in the new image corresponding to the input positions.
    """
    new_height, new_width, _ = remapping.shape
    new_positions = []

    for object_found in original_pixel_positions:
        x1, y1, x2, y2, *rest = object_found
        max_y, max_x = np.max(remapping[..., 0]), np.max(remapping[..., 1])
        min_y, min_x = np.min(remapping[..., 0]), np.min(remapping[..., 1])
        # Function to search the remapping array
        def find_new_position(orig_x, orig_y):
            distances = (remapping[..., 0] - orig_x)**2 + (remapping[..., 1] - orig_y)**2
            new_y, new_x = np.unravel_index(np.argmin(distances), distances.shape)
            return new_x, new_y
        # Map both points of the bounding box
        new_x1, new_y1 = find_new_position(int(x1), int(y1))
        new_x2, new_y2 = find_new_position(int(x2), int(y2))
        # Update the object with the new coordinates
        new_positions.append([new_x1, new_y1, new_x2, new_y2, *rest])

    return torch.tensor(new_positions)

# From DSEC github
def h5_file_to_dict(h5_file):
    h5_file = Path(h5_file)
    with h5py.File(h5_file) as fh:
        return {k: fh[k][()] for k in fh.keys()}


def yaml_file_to_dict(yaml_file):
    yaml_file = Path(yaml_file)
    with yaml_file.open() as fh:
        return yaml.load(fh, Loader=yaml.UnsafeLoader)



def conf_to_K(conf):
    K = np.eye(3)
    K[[0, 1, 0, 1], [0, 1, 2, 2]] = conf
    return K

def compute_remapping(calibration, mapping):
    mapping = mapping['rectify_map']

    K_r0 = conf_to_K(calibration['intrinsics']['camRect0']['camera_matrix'])
    K_r1 = conf_to_K(calibration['intrinsics']['camRect1']['camera_matrix'])

    R_r0_0 = np.array(calibration['extrinsics']['R_rect0'])
    R_r1_1 = np.array(calibration['extrinsics']['R_rect1'])
    R_1_0 = np.array(calibration['extrinsics']['T_10'])[:3, :3]

    # read from right to left:
    # rect. cam. 1 -> norm. rect. cam. 1 -> norm. cam. 1 -> norm. cam. 0 -> norm. rect. cam. 0 -> rect. cam. 0
    P_r0_r1 = K_r0 @ R_r0_0 @ R_1_0.T @ R_r1_1.T @ np.linalg.inv(K_r1)

    H, W = mapping.shape[:2]
    coords_hom = np.concatenate((mapping, np.ones((H, W, 1))), axis=-1)
    mapping = (np.linalg.inv(P_r0_r1) @ coords_hom[..., None]).squeeze()
    mapping = mapping[...,:2] / mapping[..., -1:]
    mapping = mapping.astype('float32')

    return mapping



def remap(labels_rgb, filename, rec_prefix):
    cam_to_cam_file = rec_prefix+f'../calibration/{filename}/calibration/cam_to_cam.yaml'
    rectification_map_file = rec_prefix+f'{filename}/events/left/rectify_map.h5'
    calibration = yaml_file_to_dict(cam_to_cam_file)
    rectification_map = h5_file_to_dict(rectification_map_file)
    remapping_map = compute_remapping(calibration, rectification_map)
    remapped_labels = []
    # for fr_label in labels_rgb:
    #     remapped_labels.append(map_pixel_positions(fr_label, remapping_map))
    remapped_labels.append(map_pixel_positions(labels_rgb, remapping_map))
    return remapped_labels 














########### YOLOX plot functions ############


def plot_ground_truth_bboxes_yoloX(image, target, grid_sizes, num_anchors_list=[5*25, 5*100, 5*400], num_classes=9, threshold=0.5):
    """
    Plots the ground-truth bounding boxes on the image for each anchor in YOLOX output.
    
    Args:
        image: The input image, usually a tensor of shape (C, H, W).
        target: Flattened tensor of shape (total_anchors, 14) containing ground-truth information.
        grid_sizes: List of grid sizes for each scale (e.g., [(13, 13), (26, 26), (52, 52)]).
        num_anchors_list: List of numbers of anchor boxes for each scale.
        num_classes: Number of classes in the dataset.
        threshold: Confidence threshold for plotting.
    """
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) format
    
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image / 10)  # Normalize image for visualization
    
    # Initialize the starting index for anchors
    start_idx = 0
    
    # Iterate over each scale (corresponding to different grid sizes)
    for scale_idx, grid_size in enumerate(grid_sizes):
        grid_size_x, grid_size_y = grid_size
        num_anchors = num_anchors_list[scale_idx]
        
        # Iterate over each anchor box for the current scale
        for anchor_idx in range(num_anchors):
            target_info = target[start_idx + anchor_idx]  # Get the flattened target for this anchor
            
            # Extract bounding box and confidence
            box = target_info[0:4]  # [x_offset, y_offset, width, height]
            confidence = target_info[4].item()  # Objectness confidence
            
            if confidence > threshold:
                # Convert normalized coordinates to image coordinates
                x_center, y_center, width, height = box
                x_center *= image.shape[1]  # Scale by image width
                y_center *= image.shape[0]  # Scale by image height
                width *= image.shape[1]  # Scale by image width
                height *= image.shape[0]  # Scale by image height
                
                # Draw the bounding box
                rect = patches.Rectangle(
                    (x_center - width / 2, y_center - height / 2), 
                    width, height, 
                    linewidth=2, 
                    edgecolor='r', 
                    facecolor='none'
                )
                ax.add_patch(rect)
        
        # Update the start index for the next scale's anchors
        start_idx += num_anchors
    
    plt.axis('off')
    plt.savefig('data_plt_ground_truth_yoloX.png')
    plt.close()

def non_max_suppression_yoloX(predictions, grid_sizes, num_anchors_list=[5*25, 5*100, 5*400], conf_threshold=0.5, iou_threshold=0.7, num_classes=9, device='cpu'):
    """
    Applies Non-Maximum Suppression (NMS) to the model predictions for all anchors and scales, without a batch dimension.
    
    Args:
        predictions: Tensor containing the model predictions. Shape: (num_anchors, 5 + num_classes)
        grid_sizes: List of grid sizes for each scale.
        num_anchors_list: List of numbers of anchor boxes for each scale.
        conf_threshold: Confidence threshold for filtering out boxes.
        iou_threshold: IoU threshold for NMS.
        num_classes: Number of classes.
        device: Device to use ('cpu' or 'cuda').
    
    Returns:
        final_boxes: Final bounding boxes after NMS.
        final_scores: Confidence scores for the boxes.
        final_classes: Predicted classes for each box.
    """
    all_boxes = []
    all_scores = []
    all_classes = []
    
    # Iterate over each scale (corresponding to a grid size and number of anchors)
    start_idx = 0
    for scale_idx, grid_size in enumerate(grid_sizes):
        num_anchors = num_anchors_list[scale_idx]
        end_idx = start_idx + num_anchors
        
        pred = predictions[start_idx:end_idx]  # Predictions for this scale
        start_idx = end_idx
        
        # Iterate over each anchor
        for anchor_idx in range(num_anchors):
            class_conf = pred[anchor_idx, 5:].sigmoid()  # Class confidences (after sigmoid)
            obj_conf = pred[anchor_idx, 4].sigmoid()     # Objectness confidence
            class_scores, class_idx = torch.max(class_conf, dim=-1)
            total_confidence = class_scores * obj_conf  # Multiply class score by objectness

            if total_confidence > conf_threshold:
                # Get the bounding box (x_center, y_center, width, height)
                box = pred[anchor_idx, 0:4].sigmoid()  # Apply sigmoid for bounding box coordinates

                # Collect the box, score, and class
                all_boxes.append(box)
                all_scores.append(total_confidence.item())
                all_classes.append(class_idx.item())
        
    # Convert lists to tensors
    all_boxes = torch.stack(all_boxes, dim=0)
    all_scores = torch.tensor(all_scores, device=device)
    all_classes = torch.tensor(all_classes, device=device)
    
    # Perform NMS
    keep = torchvision.ops.nms(all_boxes, all_scores, iou_threshold)
    
    # Apply NMS and return the final boxes, scores, and classes
    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]
    final_classes = all_classes[keep]
    
    return final_boxes, final_scores, final_classes

def plot_predictions_yoloX(image, boxes, scores, classes, threshold=0.5):
    """
    Plots the predicted bounding boxes on the image after applying NMS and considering the confidence threshold.
    
    Args:
        image: The input image (shape: H, W, C).
        boxes: Tensor of predicted bounding boxes after NMS (shape: [num_boxes, 4]).
        scores: Tensor of confidence scores for each box (shape: [num_boxes]).
        classes: Tensor of predicted class indices for each box (shape: [num_boxes]).
        threshold: Confidence threshold for filtering out boxes.
    """
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()  # Convert to numpy if it is a tensor
    
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        class_idx = classes[i]
        
        if score > threshold:
            x_center, y_center, width, height = box
            
            # Convert to pixel coordinates
            x_center = x_center * image.shape[1]
            y_center = y_center * image.shape[0]
            width = width * image.shape[1]
            height = height * image.shape[0]
            
            # Draw the bounding box
            rect = patches.Rectangle(
                (x_center - width / 2, y_center - height / 2), 
                width, height, 
                linewidth=2, 
                edgecolor='g', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Display class label and score
            ax.text(
                x_center - width / 2, 
                y_center - height / 2, 
                f"Class {class_idx}: {score:.2f}", 
                color='green', 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7)
            )
    plt.axis('off')
    plt.savefig('data_plt_boxes_yoloX.png')
    plt.close()
