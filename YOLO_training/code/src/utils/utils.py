from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import h5py
import yaml


def optimal_distance(dataset, image_type='frames'):
    if dataset == 'DDD20':
        vis_type = ['/clear', '/glare', '/night']
    elif dataset == 'DSEC':
        vis_type = ['/']
    distr_sum = np.array([], dtype=np.int64).reshape(0, 16)
    # timesteps = 0
    for v in vis_type:
        path = f'../../info_{dataset}/info_data/partial_contrast/{image_type}{v}'
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







import matplotlib.patches as patches












########### YOLOX plot functions ############



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






def generate_random_anchors(num_anchors, range_min=0.1, range_max=0.5, device='cpu'):
    """
    Generate random anchor boxes with specified ranges.

    Args:
        num_anchors: Number of anchor boxes to generate.
        range_min: Minimum value for width/height of anchors.
        range_max: Maximum value for width/height of anchors.
        device: Torch device to create the anchors on.

    Returns:
        anchor_boxes: Tensor of shape (num_anchors, 2).
    """
    return torch.rand((num_anchors, 2), device=device) * (range_max - range_min) + range_min


def reshape_yolox_predictions(predictions, batch_size, grid_shapes, prediction=False):
    """
    Reshapes YOLOX predictions into a list of tensors for each FPN level.

    Args:
        predictions (torch.Tensor): Model output of shape (batch, grids, 13).
        batch_size (int): Number of batches in the predictions.
        grid_shapes (list of tuple): List of grid shapes [(80, 60), (40, 30), (20, 15)].

    Returns:
        list of torch.Tensor: List with three elements, each of shape (batch, grid_x, grid_y, 13).
    """
    reshaped_predictions = []
    offset = 0
    for grid_x, grid_y in grid_shapes:
        grid_size = grid_x * grid_y  # Number of grids at this FPN level
        reshaped = predictions[:, offset : offset + grid_size, :].view(batch_size, grid_y, grid_x, 13)
        if prediction:
            reshaped[..., 4:] = torch.nn.functional.sigmoid(reshaped[..., 4:])

        reshaped_predictions.append(reshaped)
        offset += grid_size

    return reshaped_predictions





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

# Example: NMS function
def nms_yolox(boxes, scores, iou_threshold=0.7):
    """
    Apply Non-Maximum Suppression (NMS) on bounding boxes.
    Args:
        boxes (numpy.ndarray): Bounding boxes in the format (x1, y1, x2, y2).
        scores (numpy.ndarray): Confidence scores for each box.
        iou_threshold (float): IoU threshold for suppression.
    Returns:
        indices: Indices of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    
    return keep



def get_boxes_yolox(grid_results, grid_shapes, threshold=0.5, iou_threshold=0.7, height = 480, width = 640):
    """
    Plot bounding boxes from YOLO model predictions after applying threshold and NMS.
    Args:
        image (numpy.ndarray): The original image.
        grid_results (list): List of grid predictions from YOLO model.
        grid_shapes (list): List of grid shapes [(80, 60), (40, 30), (20, 15)].
        threshold (float): Confidence threshold for object * class confidence.
        iou_threshold (float): IoU threshold for NMS.
    """
    boxes = []
    scores = []

    for grid, shape in zip(grid_results, grid_shapes):
        grid = grid.permute(0,2,1,3).detach()
        grid = grid.reshape(*shape, -1)
        
        c_x, c_y, w, h, obj_conf = grid[..., 0], grid[..., 1], grid[..., 2], grid[..., 3], grid[..., 4]
        class_confs = grid[..., 5:]
        class_scores, _ = torch.max(class_confs, dim=-1)
        class_id = torch.argmax(class_confs, dim=-1)
        
        # Calculate scores and filter based on threshold
        scores_grid = obj_conf * class_scores
        mask = scores_grid > threshold
    

        stride_x = width / shape[0]
        stride_y = height / shape[1]
        x1 = torch.zeros_like(c_x)
        y1 = torch.zeros_like(c_y)
        x2 = torch.zeros_like(c_x)
        y2 = torch.zeros_like(c_y)

        grid_x, grid_y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")

        # Calculate the bounding box coordinates
        x1 = stride_x * grid_x + (c_x - w / 2) * 640
        y1 = stride_y * grid_y + (c_y - h / 2) * 480
        x2 = stride_x * grid_x + (c_x + w / 2) * 640
        y2 = stride_y * grid_y + (c_y + h / 2) * 480

        boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask], scores_grid[mask], class_id[mask]], axis=-1))
        scores.append(scores_grid[mask].cpu().numpy())
    
    # Concatenate results across all scales
    boxes = np.concatenate(boxes, axis=0) if boxes else np.empty((0, 4))
    scores = np.concatenate(scores, axis=0) if scores else np.empty((0,))
    # Apply NMS
    if len(boxes) > 0:
        keep = nms_yolox(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
    
    return boxes