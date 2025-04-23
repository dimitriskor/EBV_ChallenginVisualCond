import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# NMS implementation
def nms(predictions, iou_threshold):
    """Perform Non-Maximum Suppression (NMS)"""
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]
    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break

        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])

        inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        box_area = (x2 - x1) * (y2 - y1)
        union_area = box_area[current] + box_area[indices[1:]] - inter_area

        iou = inter_area / union_area

        indices = indices[1:][iou < iou_threshold]

    return predictions[keep]

# Preprocess predictions
def preprocess_predictions(grids, obj_threshold=0.5):
    """Convert grid predictions to object coordinates and filter with threshold."""
    all_boxes = []
    for grid in grids:
        cx, cy, w, h, obj_conf, *class_confs = np.split(grid, [1, 2, 3, 4, 5], axis=-1)
        class_confs = np.array(class_confs).squeeze()
        max_class_confs = class_confs.max(axis=-1)
        class_ids = class_confs.argmax(axis=-1)

        keep = obj_conf.squeeze() * max_class_confs > obj_threshold
        x1 = (cx - w / 2).squeeze()
        y1 = (cy - h / 2).squeeze()
        x2 = (cx + w / 2).squeeze()
        y2 = (cy + h / 2).squeeze()
        obj_class_conf = (obj_conf.squeeze() * max_class_confs).squeeze()

        boxes = np.stack([x1, y1, x2, y2, obj_class_conf, class_ids], axis=-1)[keep]
        all_boxes.append(boxes)

    return np.vstack(all_boxes)

# Unsupervised Metrics
def track_length(detections_by_frame, iou_threshold=0.5):
    """
    Estimate object persistence (proxy track length) without explicit tracks.
    
    Args:
        detections_by_frame (list of np.ndarray): A list where each element contains 
                                                  detections for a single frame 
                                                  with shape (num_detections, 6).
        iou_threshold (float): IoU threshold to consider two detections the same.
    
    Returns:
        float: Average track length (persistence) across all detections.
    """
    def iou(box1, box2):
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    # Initialize storage for tracks
    tracks = []
    
    for frame_idx, detections in enumerate(detections_by_frame):
        if frame_idx == 0:  # Initialize tracks with the first frame
            for det in detections:
                tracks.append([det])
            continue
        
        # Match detections to existing tracks
        matched = set()
        for track in tracks:
            last_detection = track[-1]
            best_iou = 0
            best_match = None
            
            for det_idx, detection in enumerate(detections):
                if det_idx in matched:
                    continue
                if detection[5] != last_detection[5]:  # Class mismatch
                    continue
                iou_score = iou(last_detection[:4], detection[:4])
                if iou_score > best_iou and iou_score > iou_threshold:
                    best_iou = iou_score
                    best_match = det_idx
            
            if best_match is not None:
                track.append(detections[best_match])
                matched.add(best_match)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched:
                tracks.append([detection])
    
    # Compute average track length
    track_lengths = [len(track) for track in tracks]
    return np.mean(track_lengths) if track_lengths else 0

def similarity_object_count(train_detections, test_detections):
    """Compare object counts and confidence."""
    train_count = len(train_detections)
    test_count = len(test_detections)
    return abs(train_count - test_count)

def clustering_analysis(detections):
    """Analyze the spatial clustering of (x, y, w, h)."""
    xywh = detections[:, :4]
    kmeans = KMeans(n_clusters=5, random_state=42).fit(xywh)
    return kmeans.cluster_centers_

def confidence_distribution(detections):
    """Visualize confidence score distribution."""
    confidences = detections[:, 4]
    plt.hist(confidences, bins=20, alpha=0.7)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.show()

def spatial_class_distribution(detections):
    """Visualize spatial consistency of class locations."""
    plt.figure(figsize=(10, 6))
    for class_id in np.unique(detections[:, 5]):
        class_detections = detections[detections[:, 5] == class_id]
        plt.scatter(class_detections[:, 0], class_detections[:, 1], label=f"Class {int(class_id)}", alpha=0.5)
    plt.title("Spatial Class Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Example Usage
def main():
    # Example predictions for grids of shapes (80,60), (40,30), (20,15)
    grids_train = [np.random.rand(80, 60, 8), np.random.rand(40, 30, 8), np.random.rand(20, 15, 8)]
    grids_test = [np.random.rand(80, 60, 8), np.random.rand(40, 30, 8), np.random.rand(20, 15, 8)]

    train_detections = preprocess_predictions(grids_train)
    test_detections = preprocess_predictions(grids_test)

    train_detections = nms(train_detections, 0.7)
    test_detections = nms(test_detections, 0.7)

    print("Track Length:", track_length(test_detections))
    print("Object Count Similarity:", similarity_object_count(train_detections, test_detections))
    print("Clustering Analysis:", clustering_analysis(test_detections))
    confidence_distribution(test_detections)
    spatial_class_distribution(test_detections)

if __name__ == "__main__":
    main()
