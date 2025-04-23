
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
