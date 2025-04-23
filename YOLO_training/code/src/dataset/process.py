

import torch
import cv2
import matplotlib.pyplot as plt
import torchvision
import numpy as np

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
            target[batch_idx, cell_y, cell_x, 0:4] = torch.tensor([cell_x_offset/640, cell_y_offset/480, w/640, h/480])
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





def process_labels_v8(labels, batch_size, grid_size, num_classes):
    """
    Process ground-truth labels into a target tensor compatible with the YOLO model output.

    Args:
        labels: List of ground-truth objects for each batch. Each batch is a list of objects,
                where each object is [x, y, w, h, class].
                Coordinates (x, y, w, h) are normalized to [0, 1], and `class` is an integer.
        batch_size: Number of samples in the batch.
        grid_size: The size of the grid. Dimensions are (num_of_scale, 2), where the 2nd dimension indicates the grids in x and y.
        num_classes: Number of classes in the dataset.
        bbox_per_cell: Number of bounding boxes predicted per grid cell.

    Returns:
        target: A tensor of shape (batch_size, grid_size, grid_size, bbox_per_cell * 5 + num_classes).
    """
    # Initialize the target tensor for the entire batch
    scales = len(grid_size)
    targets = []
    for grid in grid_size:
        targets.append(torch.zeros((batch_size, grid[1], grid[0], 5 + num_classes), device=labels.device))
    
    for batch_idx in range(batch_size):
        for label in labels[batch_idx]:
            x, y, w, h, o, class_idx = label
            if w*h < 0.005*640*480:
                scale = 0
            elif w*h < 0.025*640*480:
                scale = 1
            else:
                scale = 2
            class_idx = int(class_idx)
            x_cells = 640/grid_size[scale][0]
            y_cells = 480/grid_size[scale][1]
            # Determine which grid cell the object belongs to
            cell_x = int(x / x_cells)  # Column index
            cell_y = int(y / y_cells)  # Row index

            # Clamp cell indices to ensure they stay within bounds
            cell_x = min(max(cell_x, 0), grid_size[scale][0] - 1)
            cell_y = min(max(cell_y, 0), grid_size[scale][1] - 1)

            # Compute the relative coordinates within the cell
            cell_x_offset = x - cell_x * x_cells
            cell_y_offset = y - cell_y * y_cells

            # Fill in the bounding box information for the first bbox in the cell
            targets[scale][batch_idx, cell_y, cell_x, 0:4] = torch.tensor([cell_x_offset/640, cell_y_offset/480, w/640, h/480])
            if x + y + w + h != 0.0:
                targets[scale][batch_idx, cell_y, cell_x, 4] = 1.0  # Confidence score
            # Set the class label as a one-hot vector
            targets[scale][batch_idx, cell_y, cell_x, 5 + class_idx] = 1.0
    reshaped_targets = [target.view(batch_size, -1, 5 + num_classes) for target in targets]
    result = torch.cat(reshaped_targets, dim=1)
    return result
