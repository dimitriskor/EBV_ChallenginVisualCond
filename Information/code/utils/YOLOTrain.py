import pickle
import numpy as np
from ultralytics import YOLO
import norse
import torch.nn as nn
import torch
from utils.Events import Events
import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import utils.entropy as entropy
from tqdm import tqdm
from utils.utils import remap, optimal_distance
from utils.visualize import fetch_data_to_write


class YOLOLoss(nn.Module):
    def __init__(self, box_weight=0.05, obj_weight=1.0, cls_weight=0.5):
        super(YOLOLoss, self).__init__()
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.mse_loss = nn.MSELoss()      # Used for bounding box loss
        self.bce_loss = nn.BCEWithLogitsLoss()  # Used for objectness and class loss
        self.ce_loss = nn.CrossEntropyLoss()    # Optional: used for class loss

    def forward(self, predictions, targets):
        pred_boxes = predictions[..., :4]  # x, y, w, h
        pred_obj = predictions[..., 4]     # objectness score
        pred_class = predictions[..., 5:]  # class probabilities

        target_boxes = targets[..., :4]
        target_obj = targets[..., 4]
        target_class = targets[..., 5:].long()

        box_loss = self.mse_loss(pred_boxes, target_boxes)

        obj_loss = self.bce_loss(pred_obj, target_obj)

        cls_loss = self.bce_loss(pred_class, target_class.float())

        total_loss = (self.box_weight * box_loss +
                      self.obj_weight * obj_loss +
                      self.cls_weight * cls_loss)

        return total_loss
    



class LIFStateWrapper(torch.nn.Module):
    def __init__(self, lif_cell):
        super().__init__()
        self.lif_cell = lif_cell
        self.state = None  # Initialize the state

    def forward(self, x):
        x, self.state = self.lif_cell(x, self.state)  # Propagate state
        return x, self.state
    

def replace_silu_with_custom_module(model):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.SiLU):
            tau = torch.nn.Parameter(torch.tensor(0.1))
            lif_cell = norse.torch.LIFBoxCell(
                norse.torch.LIFBoxParameters(tau_mem_inv=tau, alpha=10), dt=1
            )
            setattr(model, name, LIFStateWrapper(lif_cell))  # Use the state-aware wrapper
        else:
            replace_silu_with_custom_module(layer)


def forward_recursive_with_state(module, x, state_dict=None):
    if state_dict is None:
        state_dict = {}

    for name, layer in module.named_children():
        if isinstance(layer, LIFStateWrapper):
            x, state = layer(x)
            state_dict[name] = state
        elif len(list(layer.children())) > 0:
            x, state_dict = forward_recursive_with_state(layer, x, state_dict)
        else:
            x = layer(x)

    return x, state_dict


def temporal_forward_function(module, inp):
    for x in inp:
        x, state_dict = forward_recursive_with_state(module, x)
    state_dict = {}
    return x



class HDF5Dataset(Dataset):
    def __init__(self, file_list, time_window_ms, sequence_length, rec_prefix='/mnt/raid0a/Dimitris/DSEC/event_recordings/', obj_prefix='/mnt/raid0a/Dimitris/DSEC/object_detection/'):
        self.file_list = file_list
        self.time_window_ms = time_window_ms
        self.sequence_length = sequence_length
        self.num_sequences_per_file = []
        self.rec_prefix = rec_prefix
        self.obj_prefix = obj_prefix
        self.max_bboxes = 20
        self._calculate_num_sequences()
            

    def _calculate_num_sequences(self):
        """
        Determine the number of sequences in each file without loading everything into memory.
        """
        for file_idx in self.file_list:
            file_path = self.rec_prefix + file_idx + '/events/left/events.h5'
            f = h5py.File(file_path, 'r')
            ms_to_idx = np.array(f['ms_to_idx'])
            f.close()
            self.num_sequences_per_file.append(int(len(ms_to_idx)//self.sequence_length//self.time_window_ms))

    def __len__(self):
        """
        Total number of sequences across all files.
        """
        return sum(self.num_sequences_per_file)
    
    def __getitem__(self, idx):
        cumulative_sequences = np.cumsum(self.num_sequences_per_file)
        file_idx = np.searchsorted(cumulative_sequences, idx, side='right')
        if file_idx == 0:
            sequence_idx = idx
        else:
            sequence_idx = idx - cumulative_sequences[file_idx - 1]
        file_path = self.rec_prefix + self.file_list[file_idx] + '/events/left/events.h5'
        target_path = self.obj_prefix + self.file_list[file_idx] + '/object_detections/left/tracks.npy' 
        events = Events(file_path, stack_events=self.time_window_ms)
        start_idx = sequence_idx * self.sequence_length * self.time_window_ms
        frames = events.events_to_frames_dataloader(slice=[start_idx, start_idx+self.sequence_length * self.time_window_ms])        
        target = np.load(target_path)
        target_ind = (np.array(target['t']) - int(events.file['t_offset'][()])).astype(int)  - start_idx*1000 - 1000*self.time_window_ms*self.sequence_length
        indices = np.where((target_ind >= -25000) & (target_ind <= 25000))[0]
        target = target[indices]
        target_x = target['x']/640
        target_y = target['y']/480
        target_w = target['w']/640
        target_h = target['h']/480
        target_c = target['class_id']
        target = np.stack((target_x, target_y, target_w, target_h, target_c), axis=1)
        if len(frames) < self.sequence_length:
            raise ValueError(f"Not enough frames in {file_path}")
        if target.shape[0] < self.max_bboxes:
            # Pad the target if there are fewer bounding boxes than max_bboxes
            padding = np.zeros((self.max_bboxes - target.shape[0], 5))  # 5 because we have [x, y, w, h, class_id]
            target = np.vstack([target, padding])
        frames = torch.tensor(frames)
        # frames = torch.nn.functional.interpolate(frames, size=(640, 640), mode='bilinear', align_corners=False)
        frames = frames.permute(0, 2, 3, 1)
        extra_channel = frames[:, :, :, 0].unsqueeze(-1)
        frames = torch.cat((frames, extra_channel), dim=-1)
        del events
        return frames, torch.tensor(target)





class HDF5Dataset_YOLO(Dataset):
    def __init__(self, file_list, time_window_ms, sequence_length, rec_prefix='/mnt/raid0a/Dimitris/DSEC/event_recordings/', obj_prefix='/mnt/raid0a/Dimitris/DSEC/object_detection_yolo/'):
        self.file_list = file_list
        self.time_window_ms = time_window_ms
        self.sequence_length = sequence_length
        self.num_sequences_per_file = []
        self.rec_prefix = rec_prefix
        self.obj_prefix = obj_prefix
        self.max_bboxes = 20
        self._calculate_num_sequences()
        self.optimal_distance = optimal_distance('DSEC')[0]
            

    def _calculate_num_sequences(self):
        """
        Determine the number of sequences in each file without loading everything into memory.
        """
        for file_idx in self.file_list:
            file_path = self.rec_prefix + file_idx + '/events/left/events.h5'
            f = h5py.File(file_path, 'r')
            ms_to_idx = np.array(f['ms_to_idx'])
            f.close()
            self.num_sequences_per_file.append(int(len(ms_to_idx)//self.time_window_ms//self.time_window_ms))

    def __len__(self):
        """
        Total number of sequences across all files.
        """
        return sum(self.num_sequences_per_file)
    
    def __getitem__(self, idx):
        cumulative_sequences = np.cumsum(self.num_sequences_per_file)
        file_idx = np.searchsorted(cumulative_sequences, idx, side='right')
        if file_idx == 0:
            sequence_idx = idx
        else:
            sequence_idx = idx - cumulative_sequences[file_idx - 1]
        file_path = self.rec_prefix + self.file_list[file_idx] + '/events/left/events.h5'
        target_path = self.obj_prefix + self.file_list[file_idx] + f'.pkl' 
        events = Events(file_path, stack_events=self.time_window_ms)
        start_idx = sequence_idx * self.sequence_length
        frames = events.events_to_frames_dataloader(slice=[start_idx, start_idx+self.sequence_length * self.time_window_ms])        
        with open(target_path, 'rb') as f:
            target = pickle.load(f)
        target = target[(start_idx+self.sequence_length * self.time_window_ms)//50]
        target = remap(target, self.file_list[file_idx], self.rec_prefix)[0]
        try:
            target_cx = (target[:, 0] + target[:, 2])/2
            target_w = torch.abs((target[:, 0] - target[:, 2]))
            target_cy = (target[:, 1] + target[:, 3])/2
            target_h = torch.abs(target[:, 1] - target[:, 3])
            target[:, 0] = target_cx/640
            target[:, 1] = target_cy/480
            target[:, 2] = target_w/640
            target[:, 3] = target_h/480
        except:
            target = torch.zeros((1, 6))
        metric = fetch_data_to_write(['partial_contrast'], 'info_DSEC', '',  self.file_list[file_idx])
        metric = metric['partial_contrast']
        distance = np.sum(np.square((metric-self.optimal_distance)), axis=1)
        flag = False
        if distance[(start_idx+self.sequence_length * self.time_window_ms)//50] > 0.025:
            flag = True
        if len(frames) < self.sequence_length:
            raise ValueError(f"Not enough frames in {file_path}")
        if target.shape[0] < self.max_bboxes:
            # Pad the target if there are fewer bounding boxes than max_bboxes
            padding = np.zeros((self.max_bboxes - target.shape[0], 6))  # 6 because we have [x, y, w, h, obj_conf, class_id]
            target = np.vstack([target, padding])
        frames = torch.tensor(frames)
        # frames = torch.nn.functional.interpolate(frames, size=(640, 640), mode='bilinear', align_corners=False)
        frames = frames.permute(0, 2, 3, 1)
        extra_channel = frames[:, :, :, 0].unsqueeze(-1)
        frames = torch.cat((frames, extra_channel), dim=-1)
        del events
        return frames, torch.tensor(target), flag



import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class YoloLossBase(torch.nn.Module):
    def __init__(self, grid_size, bbox_per_cell, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        super(YoloLossBase, self).__init__()
        self.grid_size = grid_size
        self.bbox_per_cell = bbox_per_cell
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, grid_size, grid_size, bbox_per_cell * 5 + num_classes)
        targets: (batch_size, 1, grid_size, grid_size, num_classes)
        """
        # Ensure both predictions and targets are on the same device
        device = predictions.device  # Get the device of the predictions tensor
        
        # Separate prediction boxes and classes
        pred_boxes = predictions[..., :self.bbox_per_cell * 5].view(
            -1, self.grid_size, self.grid_size, self.bbox_per_cell, 5  # 5 for [x, y, w, h, confidence]
        )
        pred_classes = predictions[..., self.bbox_per_cell * 5:]

        # Separate target boxes and classes
        target_boxes = targets[..., :self.bbox_per_cell*5].view(
            -1, self.grid_size, self.grid_size, self.bbox_per_cell, 5  # 5 for [x, y, w, h, confidence]
        )
        target_classes = targets[..., self.bbox_per_cell*5:]  # shape [batch, grid_size, grid_size, bbox_per_cell, num_classes]

        # We assume that the target_classes are already one-hot encoded.
        # No need for additional transformation, so we just use the one-hot target as is.
        one_hot_classes = target_classes  # Shape should already be [batch, grid_size, grid_size, bbox_per_cell, num_classes]

        # Localization loss: Only for cells with objects (confidence > 0)
        coord_mask = target_boxes[..., 4] > 0  # Mask for cells with objects
        coord_mask = coord_mask.unsqueeze(-1).expand_as(pred_boxes[..., :4])  # Expand to match pred_boxes shape

        # Apply the mask to target_boxes and pred_boxes for [x, y, w, h] comparison
        coord_loss = F.mse_loss(
            pred_boxes[..., :4][coord_mask],  # Mask applied after selecting [x, y, w, h]
            target_boxes[..., :4][coord_mask],  # Mask applied after selecting [x, y, w, h]
            reduction='sum'
        )

        # Confidence loss: Separate object and non-object cells
        obj_mask = target_boxes[..., 4] > 0  # Mask for cells with objects
        obj_mask = obj_mask.any(dim=-1)
        noobj_mask = target_boxes[..., 4] == 0  # Mask for cells without objects

        obj_conf_loss = F.mse_loss(
            pred_boxes[..., 4][obj_mask],  # Confidence values for object cells
            target_boxes[..., 4][obj_mask],  # Confidence values for object cells
            reduction='sum',
        )
        noobj_conf_loss = F.mse_loss(
            pred_boxes[noobj_mask][..., 4],  # Confidence values for non-object cells
            target_boxes[noobj_mask][..., 4],  # Confidence values for non-object cells
            reduction='sum',
        )

        # Classification loss: Use one-hot encoded classes
        class_loss = F.mse_loss(
            pred_classes[obj_mask],  # Only for cells with objects
            one_hot_classes[obj_mask],  # One-hot encoded target classes for object cells
            reduction='sum',
        )

        # Total loss
        total_loss = (
            self.lambda_coord * coord_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            class_loss
        )

        return total_loss



import torch

def process_labels(labels, batch_size, grid_size, num_classes, bbox_per_cell):
    """
    Process ground-truth labels into a target tensor compatible with the YOLO model output.

    Args:
        labels: List of ground-truth objects for each batch. Each batch is a list of objects,
                where each object is [x, y, w, h, class].
                Coordinates (x, y, w, h) are normalized to [0, 1], and `class` is an integer.
        batch_size: Number of samples in the batch.
        grid_size: The size of the grid (e.g., 7 for a 7x7 grid).
        num_classes: Number of classes in the dataset.
        bbox_per_cell: Number of bounding boxes predicted per grid cell.

    Returns:
        target: A tensor of shape (batch_size, grid_size, grid_size, bbox_per_cell * 5 + num_classes).
    """
    # Initialize the target tensor for the entire batch
    target = torch.zeros((batch_size, grid_size, grid_size, bbox_per_cell * 5 + num_classes), device=labels.device)
    
    for batch_idx in range(batch_size):
        for label in labels[batch_idx]:
            x, y, w, h, _, class_idx = label
            class_idx = int(class_idx)
            # Determine which grid cell the object belongs to
            cell_x = int(x * grid_size)  # Column index
            cell_y = int(y * grid_size)  # Row index

            # Clamp cell indices to ensure they stay within bounds
            cell_x = min(max(cell_x, 0), grid_size - 1)
            cell_y = min(max(cell_y, 0), grid_size - 1)

            # Compute the relative coordinates within the cell
            cell_x_offset = x * grid_size - cell_x
            cell_y_offset = y * grid_size - cell_y

            # Fill in the bounding box information for the first bbox in the cell
            target[batch_idx, cell_y, cell_x, 0:4] = torch.tensor([cell_x_offset, cell_y_offset, w, h])
            target[batch_idx, cell_y, cell_x, 4] = 1.0  # Confidence score

            # Set the class label as a one-hot vector
            target[batch_idx, cell_y, cell_x, bbox_per_cell * 5 + class_idx] = 1.0
    return target





def process_labels_yolox(labels, batch_size, grid_sizes, num_classes, anchor_boxes_list):
    """
    Process ground-truth labels into target tensors compatible with the YOLOX model output.
    
    Args:
        labels: List of ground-truth objects for each batch. Each batch is a list of objects,
                where each object is [x, y, w, h, class_idx].
        batch_size: Number of samples in the batch.
        grid_sizes: List of grid sizes, each corresponding to a scale of PANet.
        num_classes: Number of classes in the dataset.
        anchor_boxes_list: List of tensors, each of shape (N, 2), where N is the number of anchors
                            at a particular scale, and each anchor is [anchor_width, anchor_height].

    Returns:
        targets: A list of tensors, each corresponding to a scale, with shape:
                 (batch_size, grid_size, grid_size, N * (5 + num_classes)), where N is the number of anchor boxes.
    """
    targets = []
    
    # Process each scale (corresponding to a grid size)
    for scale_idx, grid_size in enumerate(grid_sizes):
        anchor_boxes = anchor_boxes_list[scale_idx]
        num_anchors = anchor_boxes.size(0)
        grid_size_x, grid_size_y = grid_size
        
        # Initialize the target tensor for the current scale
        target = torch.zeros(
            (batch_size, grid_size_x, grid_size_y, num_anchors, 5 + num_classes), device=labels[0].device
        )

        # Process each label in the batch
        for batch_idx in range(batch_size):
            for label in labels[batch_idx]:
                x, y, w, h, _, class_idx = label
                class_idx = int(class_idx)

                # Determine which grid cell the object belongs to
                cell_x = int(x * grid_size_x)  # Column index
                cell_y = int(y * grid_size_y)  # Row index

                # Clamp cell indices to ensure they stay within bounds
                cell_x = min(max(cell_x, 0), grid_size_x - 1)
                cell_y = min(max(cell_y, 0), grid_size_y - 1)

                # Compute the relative coordinates within the cell
                cell_x_offset = x * grid_size_x - cell_x
                cell_y_offset = y * grid_size_y - cell_y

                # Calculate the Intersection over Union (IoU) between the ground truth and anchor boxes
                anchor_ious = torch.zeros(num_anchors, device=labels[0].device)
                for anchor_idx, (anchor_w, anchor_h) in enumerate(anchor_boxes):
                    # Compute IoU between ground-truth box and this anchor box
                    intersect_w = min(w, anchor_w)
                    intersect_h = min(h, anchor_h)
                    intersect_area = intersect_w * intersect_h
                    gt_area = w * h
                    anchor_area = anchor_w * anchor_h
                    union_area = gt_area + anchor_area - intersect_area
                    iou = intersect_area / union_area
                    anchor_ious[anchor_idx] = iou

                # Find the anchor box with the highest IoU
                best_anchor_idx = torch.argmax(anchor_ious)

                # Fill in the target tensor for the best-matching anchor
                target[batch_idx, cell_y, cell_x, best_anchor_idx, 0:4] = torch.tensor(
                    [cell_x_offset, cell_y_offset, w, h], device=labels[0].device
                )
                target[batch_idx, cell_y, cell_x, best_anchor_idx, 4] = 1.0  # Confidence score

                # Set the class label as a one-hot vector
                target[batch_idx, cell_y, cell_x, best_anchor_idx, 5 + class_idx] = 1.0

        # Add the processed target for the current scale to the list
        targets.append(target)

    # Flatten each target tensor to combine across scales
    targets = [t.view(t.size(0), -1, t.size(-1)) for t in targets]
    targets = torch.cat(targets, dim=1)

    return targets


def yolox_loss(predictions, targets, num_classes=9, alpha=1.0, gamma=2.0):
    """
    Compute the YOLOX loss, including classification, objectness, and box regression losses.
    
    Args:
        predictions: List of tensors, each of shape (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes).
                     Each tensor corresponds to a different scale.
        targets: List of tensors, each with the same shape as `predictions`, containing ground-truth values for each scale.
        num_classes: Number of classes in the dataset.
        alpha: Weighting factor for classification loss (focal loss).
        gamma: Focusing parameter for focal loss.
    
    Returns:
        total_loss: Scalar tensor representing the total loss, aggregated across all scales.
    """
    total_obj_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    
    # Loop over each scale
    for scale_idx in range(len(predictions)):
        # Get predictions and targets for this scale
        pred = predictions[scale_idx]
        target = targets[scale_idx]
        
        # Separate predictions and targets into components
        pred_boxes = pred[..., 0:4]
        pred_obj = pred[..., 4]
        pred_classes = pred[..., 5:]
        
        target_boxes = target[..., 0:4]
        target_obj = target[..., 4]
        target_classes = target[..., 5:]
        
        # Objectness Loss (Binary Cross-Entropy)
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='none')
        obj_loss = obj_loss * target_obj  # Only consider positive samples
        
        # Classification Loss (Focal Loss)
        pred_classes = torch.sigmoid(pred_classes)
        focal_weight = alpha * (1 - pred_classes) ** gamma
        cls_loss = F.binary_cross_entropy(pred_classes, target_classes, reduction='none')
        cls_loss = focal_weight * cls_loss
        cls_loss = cls_loss.sum(-1) * target_obj  # Only consider positive samples
        
        # Box Regression Loss (Smooth L1 Loss)
        box_loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none')
        box_loss = box_loss.sum(-1) * target_obj  # Only consider positive samples
        
        # Sum up the losses for this scale
        total_obj_loss += obj_loss.sum()
        total_cls_loss += cls_loss.sum()
        total_box_loss += box_loss.sum()

    # Compute the total loss
    total_loss = (total_obj_loss + total_cls_loss + total_box_loss) / predictions[0].size(0)  # Normalize by batch size

    return total_loss


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
