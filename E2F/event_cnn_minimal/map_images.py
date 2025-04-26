import os
import numpy as np
import cv2
import h5py
import yaml
import torch
import hdf5plugin
from pathlib import Path


def h5_file_to_dict(h5_file):
    h5_file = Path(h5_file)
    with h5py.File(h5_file, 'r') as fh:
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

    P_r0_r1 = K_r0 @ R_r0_0 @ R_1_0.T @ R_r1_1.T @ np.linalg.inv(K_r1)

    H, W = mapping.shape[:2]
    coords_hom = np.concatenate((mapping, np.ones((H, W, 1))), axis=-1)
    mapping = (np.linalg.inv(P_r0_r1) @ coords_hom[..., None]).squeeze()
    mapping = mapping[..., :2] / mapping[..., -1:]
    mapping = mapping.astype('float32')

    return mapping

def remap_image(image, remapping_map):
    """
    Remaps the entire image based on the remapping array.
    
    Args:
        image (np.ndarray): The original image.
        remapping_map (np.ndarray): The remapping array of shape (new_height, new_width, 2),
                                    where each entry gives (orig_x, orig_y).
    
    Returns:
        np.ndarray: The remapped image.
    """
    new_height, new_width = remapping_map.shape[:2]
    
    # Extract the x and y mapping coordinates
    map_x = remapping_map[..., 0].astype(np.float32)
    map_y = remapping_map[..., 1].astype(np.float32)

    # Apply the remapping to the image
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return remapped_image

def remap_full_image(image, filename, rec_prefix):
    """
    Loads calibration and rectification data, computes remapping, and applies it to an image.
    
    Args:
        image (np.ndarray): The original image to be remapped.
        filename (str): The file identifier for calibration data.
        rec_prefix (str): The root path to the calibration and rectification files.
    
    Returns:
        np.ndarray: The remapped image.
    """
    cam_to_cam_file = rec_prefix + f'../calibration/{filename}/calibration/cam_to_cam.yaml'
    rectification_map_file = rec_prefix + f'{filename}/events/left/rectify_map.h5'
    
    calibration = yaml_file_to_dict(cam_to_cam_file)
    rectification_map = h5_file_to_dict(rectification_map_file)
    remapping_map = compute_remapping(calibration, rectification_map)
    
    return remap_image(image, remapping_map)


rec_prefix ='/mnt/raid0a/Dimitris/DSEC/images_recordings/'
rec_names = os.listdir(rec_prefix)
for name in rec_names:
    full_prefix = rec_prefix+name+'/images/left/rectified/'
    files = os.listdir(full_prefix)
    try:
        os.makedirs(f'mapped_images/{name}')
    except:
        pass
    for image in files:
        img = cv2.imread(full_prefix+image)
        remapped_img = remap_full_image(img, name, rec_prefix+'../event_recordings/')
        cv2.imwrite(f'mapped_images/{name}/{image}', remapped_img)
        print(remapped_img.shape)