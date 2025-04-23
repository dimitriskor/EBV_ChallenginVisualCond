from pathlib import Path
import numpy as np
import torch
import h5py
import yaml

def fetch_data_to_write(plot_info, dataset, save_type, filename):
    if plot_info == None:
        pass
    elif plot_info == ['all']:
        data_path_entropy = f'../../{dataset}/info_data/entropy/frames{save_type}/{filename.split("/")[-1]}.npy'
        data_path_partial_contr = f'../../{dataset}/info_data/partial_contrast/frames{save_type}/{filename.split("/")[-1]}.npy'
        data_path_std_c = f'../../{dataset}/info_data/std_C/frames{save_type}/{filename.split("/")[-1]}.npy'
        data_entropy = np.load(data_path_entropy)
        data_partial_contr = np.load(data_path_partial_contr)
        data_std_C = np.load(data_path_std_c)
        data = [data_entropy, data_partial_contr, data_std_C]
    else:
        data = {}
        for info in plot_info:
            data_path = f'../../{dataset}/info_data/{info}/frames{save_type}/{filename.split("/")[-1]}.npy'
            data[info] = np.load(data_path)
    return data





def map_pixel_positions(original_pixel_positions, remapping):
    """
    Maps specific pixel positions from the original image to their corresponding
    positions in the new image using the remapping array.

    Args:
        original_pixel_positions: List of [x1, y1, x2, y2, ...] objects, where x1, y1 and x2, y2
                                  are pixel indices in the original image.
        remapping: The remapping array of shape (new_height, new_width, 2).
                   Each entry remapping[new_y, new_x] gives the (orig_x, orig_y) that maps to (new_x, new_y).

    Returns:
        A list of updated pixel positions in the new image corresponding to the input positions.
    """
    new_height, new_width, _ = remapping.shape
    new_positions = []

    for object_found in original_pixel_positions:
        x1, y1, x2, y2, *rest = object_found
        max_y, max_x = np.max(remapping[..., 0]), np.max(remapping[..., 1])
        min_y, min_x = np.min(remapping[..., 0]), np.min(remapping[..., 1])
        # Function to search the remapping array
        def find_new_position(orig_x, orig_y):
            distances = (remapping[..., 0] - orig_x)**2 + (remapping[..., 1] - orig_y)**2
            new_y, new_x = np.unravel_index(np.argmin(distances), distances.shape)
            return new_x, new_y
        # Map both points of the bounding box
        new_x1, new_y1 = find_new_position(int(x1), int(y1))
        new_x2, new_y2 = find_new_position(int(x2), int(y2))
        # Update the object with the new coordinates
        new_positions.append([new_x1, new_y1, new_x2, new_y2, *rest])

    return torch.tensor(new_positions)

# From DSEC github
def h5_file_to_dict(h5_file):
    h5_file = Path(h5_file)
    with h5py.File(h5_file) as fh:
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

    # read from right to left:
    # rect. cam. 1 -> norm. rect. cam. 1 -> norm. cam. 1 -> norm. cam. 0 -> norm. rect. cam. 0 -> rect. cam. 0
    P_r0_r1 = K_r0 @ R_r0_0 @ R_1_0.T @ R_r1_1.T @ np.linalg.inv(K_r1)

    H, W = mapping.shape[:2]
    coords_hom = np.concatenate((mapping, np.ones((H, W, 1))), axis=-1)
    mapping = (np.linalg.inv(P_r0_r1) @ coords_hom[..., None]).squeeze()
    mapping = mapping[...,:2] / mapping[..., -1:]
    mapping = mapping.astype('float32')

    return mapping



def remap(labels_rgb, filename, rec_prefix):
    cam_to_cam_file = rec_prefix+f'../calibration/{filename}/calibration/cam_to_cam.yaml'
    rectification_map_file = rec_prefix+f'{filename}/events/left/rectify_map.h5'
    calibration = yaml_file_to_dict(cam_to_cam_file)
    rectification_map = h5_file_to_dict(rectification_map_file)
    remapping_map = compute_remapping(calibration, rectification_map)
    remapped_labels = []
    # for fr_label in labels_rgb:
    #     remapped_labels.append(map_pixel_positions(fr_label, remapping_map))
    remapped_labels.append(map_pixel_positions(labels_rgb, remapping_map))
    return remapped_labels 



