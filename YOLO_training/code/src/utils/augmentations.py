import numpy as np
import random
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F

def augment_video_with_scale_and_translation(frames, objects, scale_range=(0.75, 1.5), rotation_range=(-15, 15), translation_range=(-100, 100)):
    """
    Perform data augmentation on video frames and objects.

    Args:
        frames (np.ndarray): Video frames of shape [time, height, width, channels].
        objects (np.ndarray): Object tensor of shape [time, obj, [c_x, c_y, w, h]].
        scale_range (tuple): Range of scaling factors for resizing.
        rotation_range (tuple): Range of rotation angles in degrees.
        translation_range (tuple): Range of translation offsets in pixels.
    
    Returns:
        np.ndarray: Augmented frames.
        np.ndarray: Augmented objects.
    """
    batch, height, width, channels = frames.shape
    augmented_frames = []
    augmented_objects = []
    scale = random.uniform(*scale_range)
    tx = random.randint(*translation_range)
    ty = random.randint(*translation_range)

    for t in range(batch):
        # Load frame and objects
        frame = frames
        objs = objects
        
        # Convert frame to PIL image for geometric transformations
        frame_pil = Image.fromarray(frame)
        original_width, original_height = frame_pil.size

        # 1. Scaling
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        frame_pil = frame_pil.resize((new_width, new_height), Image.ANTIALIAS)
        objs[:, :2] *= scale  # Scale center coordinates
        objs[:, 2:] *= scale  # Scale width and height


        # 3. Translation
        frame_pil = frame_pil.transform(frame_pil.size, Image.AFFINE, (1, 0, tx, 0, 1, ty))
        objs[:, 0] += tx/640
        objs[:, 1] += ty/480

        # Convert back to numpy and add to augmented frames
        augmented_frame = np.array(frame_pil)
        augmented_frames.append(augmented_frame)
        augmented_objects.append(objs)
    
    return torch.tensor(augmented_frames), augmented_objects


def zoom_translate_crop(frame, objects, relaxation, scale_range=(0.8, 1.2), rotation_range=(-15, 15), translation_range=(-1, 1)):
    """
    Zooms into the frame and translates it while keeping the final frame size unchanged.
    Excess areas are cropped.

    Args:
        frame (torch.Tensor): The input frame of shape (C, H, W).
        objects (torch.Tensor): Tensor of object coordinates of shape (N, 4),
                                where each row is (x_center, y_center, width, height).
        scale (float): Scaling factor (>1 for zoom in, <1 for zoom out).
        translation (tuple): Translation (tx, ty) in pixels.

    Returns:
        torch.Tensor: Transformed frame of same shape as the original.
        torch.Tensor: Transformed object coordinates.
    """
    frame = torch.as_tensor(frame)  # Ensure it's a PyTorch tensor
    H, W, C = frame.shape  # Get original shape

    # Random scale factor
    scale = random.uniform(*scale_range)

    # Random translation offsets
    tx = random.randint(*translation_range)
    ty = random.randint(*translation_range)

    # Compute new scaled dimensions
    new_H, new_W = int(H * scale), int(W * scale)

    # Resize the frame with bilinear interpolation
    frame_resized = F.interpolate(frame.permute(2, 0, 1).unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)

    # Compute cropping coordinates after translation
    x1, y1 = max(0, -tx), max(0, -ty)
    x2, y2 = min(new_W, W - tx), min(new_H, H - ty)

    # Ensure cropping doesn't exceed bounds
    frame_cropped = frame_resized[y1:y1 + H, x1:x1 + W, :]

    # Compute padding if needed (to maintain original size)
    pad_h = max(0, H - frame_cropped.shape[0])
    pad_w = max(0, W - frame_cropped.shape[1])
    # print(frame.shape)
    # Pad to restore original size
    frame_final = F.pad(frame_cropped.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode="constant", value=0).permute(1, 2, 0)
    # print(frame_final.shape)
    # Transform object coordinates
    # print(objects)
    if objects.dim() == 2:
        objects_transformed = objects.clone()
        # print(objects[:, 0], objects[:, 0] * scale, (objects[:, 0] * scale) - tx/640)
        # try:
        objects_transformed[:, 1] = (objects[:, 1] * scale) + tx/640  # Scale & translate x_center
        objects_transformed[:, 2] = (objects[:, 2] * scale) + ty/480  # Scale & translate y_center
        objects_transformed[:, 3] *= scale  # Scale width
        objects_transformed[:, 4] *= scale  # Scale height

        x1 = objects_transformed[:, 1] - objects_transformed[:, 3] / 2  # x1 = x_center - width / 2
        x2 = objects_transformed[:, 1] + objects_transformed[:, 3] / 2  # x2 = x_center + width / 2
        y1 = objects_transformed[:, 2] - objects_transformed[:, 4] / 2  # y1 = y_center - height / 2
        y2 = objects_transformed[:, 2] + objects_transformed[:, 4] / 2  # y2 = y_center + height / 2
        x1 = torch.clamp(x1, min=0.0, max=1.0)
        x2 = torch.clamp(x2, min=0.0, max=1.0)
        y1 = torch.clamp(y1, min=0.0, max=1.0)
        y2 = torch.clamp(y2, min=0.0, max=1.0)
        # print(x1, x2, y1, y2)

        objects_transformed[:, 1] = (x2 + x1)/2
        objects_transformed[:, 2] = (y2 + y1)/2
        objects_transformed[:, 3] = x2 - x1
        objects_transformed[:, 4] = y2 - y1
        objects_transformed = objects_transformed[objects_transformed[:, 3] > relaxation]
        objects_transformed = objects_transformed[objects_transformed[:, 4] > relaxation]
        objects = objects_transformed

    # except IndexError:
    #     objects_transformed = torch.empty(0, 5)
    #     pass
    # print(objects_transformed)
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # print(objects_transformed)
    # plt.imshow((frame_final[:,:,0]))  # This is your image
    # ax = plt.gca()  # Get the current axes
    # for obj in objects_transformed:
    #     _, x, y, w, h = obj
    #     x1 = x - w / 2
    #     y1 = y - h / 2
    #     x2 = x + w / 2
    #     y2 = y + h / 2
        
    #     x1, y1, x2, y2 = x1*640, y1*480, x2*640, y2*480
    #     w, h = w*640, h*480
    #     # Create a rectangle to represent the bounding box
    #     rect = patches.Rectangle((x1, y1), w, h,
    #                             linewidth=2, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)

    # plt.savefig('fsdf.png')  #
    # sdv
    return frame_final, objects




def augment_video_with_flip_and_rotation(frame, object):
    """
    Perform data augmentation on video frames and objects with flips and 90-degree rotations.

    Args:
        frames (np.ndarray): Video frames of shape [time, height, width, channels].
        objects (np.ndarray): Object tensor of shape [time, obj, [c_x, c_y, w, h]].

    Returns:
        np.ndarray: Augmented frames.
        np.ndarray: Augmented objects.
    """
    height, width, channels = frame.shape
    augmented_frames = []
    augmented_objects = []

    transformations = random.choices(
        ["None", "horizontal_flip", "vertical_flip"],
        k=random.randint(1, 2)  # Apply between 1 and 3 transformations
    )

    # Load frame and objects
    objs = object # Copy to avoid modifying original data

    # Randomly apply one or more transformations
    try:
        frame = frame.numpy()
    except:
        pass
    for transform in transformations:
        if transform == "horizontal_flip":
            # Flip the frame horizontally
            frame = np.flip(frame, axis=1)
            try:
                objs[:, 1] = 1 - objs[:, 1]  # Adjust x-coordinates
            except:
                pass
        elif transform == "vertical_flip":
            # Flip the frame vertically
            frame = np.flip(frame, axis=0)
            try:
                objs[:, 2] = 1 - objs[:, 2]  # Adjust y-coordinates
            except:
                pass
        elif transform == "None":
            continue

    return torch.tensor(frame.copy()), objs




def silence_channels(frame, channels = 3):
    num_channels_to_zero = np.random.choice([i for i in range(1, channels+1)])
    channels_to_zero = np.random.choice(10, num_channels_to_zero, replace=False)
    frame[:, :, channels_to_zero] = 0
    return frame




    ### Rotational transformations
        # elif transform == "rotate_90":
        #     # Rotate the frame 90 degrees clockwise
        #     frame_pil = frame_pil.transpose(Image.ROTATE_270)
        #     objs[:, [0, 1]] = objs[:, [1, 0]]  # Swap x and y
        #     objs[:, 0] = 1 - objs[:, 0]  # Adjust new x-coordinates

        # elif transform == "rotate_180":
        #     # Rotate the frame 180 degrees
        #     frame_pil = frame_pil.transpose(Image.ROTATE_180)
        #     objs[:, 0] = 1 - objs[:, 0]  # Adjust x-coordinates
        #     objs[:, 1] = 1 - objs[:, 1]  # Adjust y-coordinates

        # elif transform == "rotate_270":
        #     # Rotate the frame 270 degrees clockwise
        #     frame_pil = frame_pil.transpose(Image.ROTATE_90)
        #     objs[:, [0, 1]] = objs[:, [1, 0]]  # Swap x and y
        #     objs[:, 1] = 1 - objs[:, 1]  # Adjust new y-coordinates

        # Convert back to numpy and add to augmented frames