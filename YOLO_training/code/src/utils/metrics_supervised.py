import numpy as np
from sklearn.metrics import f1_score
import torch
from torchvision.ops import nms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.utils import get_boxes_yolox


def compute_iou(boxes1, boxes2):
    # Compute pairwise IoU
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union

def compute_metrics_for_frames(pred_boxes_list, target_boxes_list, iou_thresholds=[0.05 * (1+i) for i in range(19)], conf_thres=0.5, num_classes=8):
    """
    Compute metrics for multiple frames while accounting for classes across all frames.
    
    Args:
        pred_boxes_list (list of numpy.ndarray): List of predicted boxes for each frame.
        target_boxes_list (list of numpy.ndarray): List of ground truth boxes for each frame.
        iou_thresholds (list of float): IoU thresholds for AP calculation.
        conf_thres (float): Confidence threshold for predictions.
        num_classes (int): Number of classes in the dataset.
    
    Returns:
        metrics (dict): Dictionary containing mAP50, mAP, and F1 metrics.
    """
    metrics = {
        "mAP50": 0,
        "mAP": 0,
        "F1": 0,
        "F1_class": [],
        "mAP_thres": []
    }
    aps = []
    f1_scores_mean = []
    IoUs = []

    # Initialize accumulators for per-class TP, FP, FN
    class_tp = np.zeros(num_classes, dtype=np.float32)
    class_fp = np.zeros(num_classes, dtype=np.float32)
    class_fn = np.zeros(num_classes, dtype=np.float32)
    
    # Iterate over IoU thresholds
    for t in iou_thresholds:
        class_aps_all_frames = [[] for _ in range(num_classes)]  # Accumulate APs for each class across frames
        
        # Accumulate metrics across all frames
        for frame_idx, (pred_boxes, target_boxes) in enumerate(zip(pred_boxes_list, target_boxes_list)):
            for c in range(num_classes):
                pred_mask = pred_boxes[:, -1] == c
                target_mask = target_boxes[:, -1] == c
                
                pred_for_class = pred_boxes[pred_mask]
                target_for_class = target_boxes[target_mask]
                
                if len(pred_for_class) == 0 and len(target_for_class) == 0:
                    # No predictions or targets for this class; skip this frame for this class
                    continue
                
                # Get pairwise IoU for the current class
                if len(pred_for_class) == 0 or len(target_for_class) == 0:
                    matched_preds = np.zeros(len(pred_for_class), dtype=bool)
                    matched_targets = np.zeros(len(target_for_class), dtype=bool)
                else:
                    iou_class = compute_iou(pred_for_class[:, :4], target_for_class[:, :4])
                    iou_values = iou_class.flatten()
                    iou_values = iou_values[np.isfinite(iou_values)]
                    if t == 0.05:
                       IoUs += iou_values.tolist()
                    
                    # Match predictions to targets
                    matched_preds = np.zeros(len(pred_for_class), dtype=bool)
                    matched_targets = np.zeros(len(target_for_class), dtype=bool)
                    
                    for pred_idx, iou_row in enumerate(iou_class):
                        max_iou_idx = np.argmax(iou_row)
                        if iou_row[max_iou_idx] > t and not matched_targets[max_iou_idx]:
                            matched_preds[pred_idx] = True
                            matched_targets[max_iou_idx] = True
                
                # Compute TP, FP, FN
                tp = matched_preds.sum()
                fp = (~matched_preds).sum()
                fn = (~matched_targets).sum()
                
                # Accumulate per-class metrics
                if t == 0.5:
                    class_tp[c] += tp
                    class_fp[c] += fp
                    class_fn[c] += fn
                
                # Compute AP
                if tp + fp + fn > 0:
                    sorted_indices = np.argsort(-pred_for_class[:, 4])  # Sort by confidence
                    sorted_tp = matched_preds[sorted_indices].astype(int)
                    cum_tp = np.cumsum(sorted_tp)
                    cum_fp = np.cumsum(~sorted_tp)
                    
                    recalls = cum_tp / (tp + fn) if (tp + fn) > 0 else np.zeros_like(cum_tp)
                    precisions = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp).all() else np.zeros_like(cum_tp)

                    ap = 0
                    if len(recalls) > 1:  # Ensure there are at least two points for AP computation
                        ap = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
                    class_aps_all_frames[c].append(ap)
        
        # Compute mean AP per class across all frames where the class exists
        class_aps = [np.mean(class_aps_all_frames[c]) if class_aps_all_frames[c] else 0 for c in range(num_classes)]
        aps.append(np.mean(class_aps))  # Average AP across classes
    
    # Calculate F1 score per class using accumulated TP, FP, FN
    f1_per_class = []
    for c in range(num_classes):
        precision = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) > 0 else 0
        recall = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)
    
    metrics["mAP50"] = aps[9] if len(iou_thresholds) == 19 else np.mean(aps)
    metrics["mAP"] = np.mean(aps[9:])
    metrics["F1"] = np.mean(f1_per_class)
    metrics["F1_class"] = f1_per_class  # Add F1 scores for each class
    metrics["mAP_thres"] =  aps
    return metrics, IoUs
