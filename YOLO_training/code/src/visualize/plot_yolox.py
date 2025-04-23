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



def plot_results_yolox(image, grid_results, grid_shapes, threshold=0.5, iou_threshold=0.7):
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
    image = image[0].permute(1, 2, 0).cpu()
    height, width = image.shape[:2]  # Image dimensions
    image = torch.sum(image, dim=-1)
    for grid, shape in zip(grid_results, grid_shapes):
        grid = grid.permute(0,2,1,3)[0]
        grid = grid.reshape(*shape, -1)
        
        c_x, c_y, w, h, obj_conf = grid[..., 0], grid[..., 1], grid[..., 2], grid[..., 3], grid[..., 4]
        class_confs = grid[..., 5:]
        class_scores, _ = torch.max(class_confs, axis=-1)
        
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

        boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=-1))
        scores.append(scores_grid[mask].cpu().numpy())
    
    # Concatenate results across all scales
    boxes = np.concatenate(boxes, axis=0) if boxes else np.empty((0, 4))
    scores = np.concatenate(scores, axis=0) if scores else np.empty((0,))
    # Apply NMS
    if len(boxes) > 0:
        keep = nms_yolox(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
    
    # Plot image and boxes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", facecolor="none", linewidth=2)
        ax.add_patch(rect)
    
    plt.show()
    plt.savefig("test.png")
