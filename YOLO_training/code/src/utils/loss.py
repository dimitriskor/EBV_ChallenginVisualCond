import torch
import torch.nn as nn
import torch.nn.functional as F


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
    



class YoloLossBase(torch.nn.Module):
    def __init__(self, grid_size, bbox_per_cell, num_classes, lambda_coord=5.0, lambda_noobj=0.1):
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






class YoloLossBase_Yolov8(nn.Module):
    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.1):
        super(YoloLossBase_Yolov8, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, grid[1]*grid[0], 13)
        targets: (batch_size, grid[1]*grid[0], 13)
        """
        # Ensure both predictions and targets are on the same device
        device = predictions.device
        # predictions = torch.sigmoid(predictions)
        # Separate components
        pred_boxes = predictions[..., :4]  # [cx, cy, w, h]
        pred_conf = torch.sigmoid(predictions[..., 4])   # Confidence
        pred_classes = torch.sigmoid(predictions[..., 5:])  # Class probabilities (8 classes)
        # pred_conf = torch.sigmoid(predictions[..., 4])   # Confidence
        # pred_classes = torch.sigmoid(predictions[..., 5:])  # Class probabilities (8 classes)

        target_boxes = targets[..., :4]  # [cx, cy, w, h]
        target_conf = targets[..., 4]   # Confidence
        target_classes = targets[..., 5:]  # One-hot encoded class probabilities

        # Localization loss: Only for cells with objects (confidence > 0)
        coord_mask = target_conf > 0  # Mask for cells with objects

        # Apply the mask for [cx, cy, w, h]
        coord_loss = F.mse_loss(
            pred_boxes[coord_mask],  # [cx, cy, w, h]
            target_boxes[coord_mask],
            # reduction='sum',
        )

        # Confidence loss: Separate object and non-object cells
        obj_mask = target_conf > 0  # Mask for cells with objects
        noobj_mask = target_conf == 0  # Mask for cells without objects

        obj_conf_loss = F.mse_loss(
            pred_conf[obj_mask],  # Confidence values for object cells
            target_conf[obj_mask],
            # reduction='sum',
        )
        noobj_conf_loss = F.mse_loss(
            pred_conf[noobj_mask],  # Confidence values for non-object cells
            target_conf[noobj_mask],
            # reduction='sum',
        )

        # Classification loss: Only for cells with objects
        class_loss = F.mse_loss(
            pred_classes[obj_mask],  # Predicted class probabilities
            target_classes[obj_mask],  # One-hot encoded target classes
            # reduction='sum',
        )

        # Total loss
        total_loss = (
            self.lambda_coord * coord_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            class_loss
        )

        return total_loss



def yolox_loss(predictions, targets, num_classes=8, alpha=1.0, gamma=2.0):
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







class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes, lambda_obj=5.0, lambda_cls=1.0, lambda_box=1.0, lambda_l1=0.05, l_noobj=2):
        super(YOLOv8Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_l1 = lambda_l1
        self.l_noobj = l_noobj

        # Loss components
        self.bce_objectness = nn.BCEWithLogitsLoss()
        self.bce_class = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, preds, targets):
        """
        Compute YOLOv8 loss.

        Args:
            preds: Predicted tensor of shape [batch, grid, 5 + num_classes].
            targets: Target tensor of shape [batch, grid, 5 + num_classes].

        Returns:
            total_loss: Total loss (scalar).
        """
        # Split predictions and targets
        pred_bbox = preds[..., :4]     # Bounding box predictions
        pred_objectness = preds[..., 4]  # Objectness logits
        pred_class = preds[..., 5:]     # Class logits
        
        target_bbox = targets[..., :4]     # Ground truth bounding boxes
        target_objectness = targets[..., 4]  # Objectness labels
        target_class = targets[..., 5:]     # One-hot encoded class labels

        # Objectness loss

        # Classification loss
        pos_mask = target_objectness > 0  # Binary mask where 1 = object present, 0 = no object
        loss_noobj = self.bce_objectness(pred_objectness[~pos_mask], target_objectness[~pos_mask])
        if torch.sum(pos_mask) != 0:
            loss_obj = self.bce_objectness(pred_objectness[pos_mask], target_objectness[pos_mask])
            loss_cls = self.bce_class(pred_class[pos_mask], target_class[pos_mask])
        # Localization loss
            iou_loss = 1.0 - self.compute_ciou(pred_bbox[pos_mask], target_bbox[pos_mask])
            l1_loss = self.l1_loss(pred_bbox[pos_mask], target_bbox[pos_mask])
            loss_box = iou_loss + self.lambda_l1 * l1_loss
        #     loss_box = F.mse_loss(
        #     pred_bbox[pos_mask],  # Predicted class probabilities
        #     target_bbox[pos_mask],  # One-hot encoded target classes
        #     reduction='sum',
        # )

        # Total loss
            total_loss = (
                self.lambda_obj * loss_obj +
                self.l_noobj * loss_noobj +
                self.lambda_cls * loss_cls +
                self.lambda_box * loss_box.mean()
            )
        else:
            total_loss = (
                self.l_noobj * loss_noobj
            )

        
        return total_loss

    def compute_ciou(self, pred_boxes, target_boxes):
        """
        Compute Complete IoU (CIoU) between predicted and target bounding boxes.

        Args:
            pred_boxes: Predicted bounding boxes, shape [batch, grid, 4].
                        Each box is represented as [x, y, w, h] (center-x, center-y, width, height).
            target_boxes: Target bounding boxes, shape [batch, grid, 4].
                          Same format as pred_boxes.

        Returns:
            ciou: CIoU values for each bounding box, shape [batch, grid].
        """
        # Extract box parameters
        pred_x, pred_y, pred_w, pred_h = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
        target_x, target_y, target_w, target_h = target_boxes[..., 0], target_boxes[..., 1], target_boxes[..., 2], target_boxes[..., 3]

        # Compute IoU
        pred_x1, pred_y1 = pred_x - pred_w / 2, pred_y - pred_h / 2  # Top-left corner of predicted box
        pred_x2, pred_y2 = pred_x + pred_w / 2, pred_y + pred_h / 2  # Bottom-right corner of predicted box
        target_x1, target_y1 = target_x - target_w / 2, target_y - target_h / 2
        target_x2, target_y2 = target_x + target_w / 2, target_y + target_h / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-7)

        # Compute center distance
        center_dist = (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2

        # Compute enclosing box diagonal length
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

        # Compute aspect ratio term
        v = (4 / (torch.pi ** 2)) * (torch.atan(target_w / (target_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7))) ** 2
        alpha = v / (1 - iou + v + 1e-7)

        # Compute CIoU
        ciou = iou - (center_dist / (enclose_diag + 1e-7)) - (alpha * v)

        return ciou