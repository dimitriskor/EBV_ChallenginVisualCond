import os
import matplotlib.pyplot as plt
import cv2
from utils.Events import Events
from utils.Frames import Frames
from tqdm import tqdm
import numpy as np
import utils.utils as utils



def write_info_in_frame_partial_contrast(frame, data_partial, pos_x, pos_y, name, rotate):
    opt_distr = np.array([0.3, 0.6, 0.07, 0.015, 0.015])
    pos_y_ = pos_y
    # for idx, data in enumerate(data_partial):
    #     data = data - opt_distr[idx]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.4
    #     color = (0,)  # White color for grayscale (single channel)
    #     thickness = 1
    #     text = f'{name} {idx}: {data:.3f}'
    #     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    #     canvas = np.zeros((text_height + baseline, text_width), dtype=np.uint8)+255
    #     cv2.putText(canvas, text, (0, text_height), font, font_scale, color, thickness)
    #     canvas_rotated = canvas
    #     if rotate:
    #         canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_180)
    #         canvas_rotated = cv2.flip(canvas_rotated, 1)
    #     height, width = frame.shape[:2]

    #     # Define the position for the text (top right corner)
    #     x_pos = width - text_width - 10  # Slight padding from the edge
    #     y_pos = pos_y_

    #     # Place the rotated text on the original image
    #     y1, y2 = y_pos, y_pos + canvas_rotated.shape[0]
    #     x1, x2 = x_pos, x_pos + canvas_rotated.shape[1]

    #     # Ensure the text fits within the image dimensions
    #     if y2 <= height and x2 <= width:
    #         frame[int(y1):int(y2), int(x1):int(x2), 0] = canvas_rotated
    #         frame[int(y1):int(y2), int(x1):int(x2), 1] = canvas_rotated
    #         frame[int(y1):int(y2), int(x1):int(x2), 2] = canvas_rotated
    #     pos_y_ -= 30

    dist = np.sum(np.square((data_partial-opt_distr)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0,)  # White color for grayscale (single channel)
    thickness = 1
    text = f'{name}: {dist:.3f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    canvas = np.zeros((text_height + baseline, text_width), dtype=np.uint8)+255
    cv2.putText(canvas, text, (0, text_height), font, font_scale, color, thickness)
    canvas_rotated = canvas
    if rotate:
        canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_180)
        canvas_rotated = cv2.flip(canvas_rotated, 1)    
    height, width = frame.shape[:2]

    # Define the position for the text (top right corner)
    x_pos = width - text_width - pos_x  # Slight padding from the edge
    y_pos = pos_y

    # Place the rotated text on the original image
    y1, y2 = y_pos, y_pos + canvas_rotated.shape[0]
    x1, x2 = x_pos, x_pos + canvas_rotated.shape[1]

    # Ensure the text fits within the image dimensions
    if y2 <= height and x2 <= width:
        frame[int(y1):int(y2), int(x1):int(x2), 0] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 1] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 2] = canvas_rotated

    return frame



def write_info_in_frame(frame, data, pos_x, pos_y, name, rotate):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0,)  # White color for grayscale (single channel)
    thickness = 1
    text = f'{name}: {data:.3f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    canvas = np.zeros((text_height + baseline, text_width), dtype=np.uint8)+255
    cv2.putText(canvas, text, (0, text_height), font, font_scale, color, thickness)
    canvas_rotated = canvas
    if rotate:
        canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_180)
        canvas_rotated = cv2.flip(canvas_rotated, 1)    
    height, width = frame.shape[:2]

    # Define the position for the text (top right corner)
    x_pos = width - text_width - pos_x  # Slight padding from the edge
    y_pos = pos_y

    # Place the rotated text on the original image
    y1, y2 = y_pos, y_pos + canvas_rotated.shape[0]
    x1, x2 = x_pos, x_pos + canvas_rotated.shape[1]

    # Ensure the text fits within the image dimensions
    if y2 <= height and x2 <= width:
        frame[int(y1):int(y2), int(x1):int(x2), 0] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 1] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 2] = canvas_rotated

    return frame




def write_in_frame(frame, plot_info, data, idx, rotate):
    height, width = frame.shape[:2]
    if plot_info != None and plot_info != 'all':
        number_of_spaces = len(data) + 1
        for id, data_key in enumerate(data.keys()):
            if data_key == 'partial_contrast':
                frame = write_info_in_frame_partial_contrast(frame, data[data_key][idx], (1+id)*height/number_of_spaces, 10, data_key, rotate)
            else:
                frame = write_info_in_frame(frame, data[data_key][idx], (1+id)*height/number_of_spaces, 10, data_key, rotate)

    if plot_info == ['all']:
        d_entropy = data[0][idx]
        d_partial_contr = data[1][idx]
        d_std_C = data[2][idx]
        frame = write_info_in_frame(frame, d_entropy, height/4, 10, 'entr', rotate)
        frame = write_info_in_frame_partial_contrast(frame, d_partial_contr, 2*height/4, 10, 'part_C', rotate)
        frame = write_info_in_frame(frame, d_std_C, 3*height/4, 10, 'std_C', rotate)
    return frame


def fetch_data_to_write(plot_info, dataset, save_type, filename):
    if plot_info == None:
        pass
    elif plot_info == ['all']:
        data_path_entropy = f'../{dataset}/info_data/entropy/frames{save_type}/{filename.split("/")[-1]}.npy'
        data_path_partial_contr = f'../{dataset}/info_data/partial_contrast/frames{save_type}/{filename.split("/")[-1]}.npy'
        data_path_std_c = f'../{dataset}/info_data/std_C/frames{save_type}/{filename.split("/")[-1]}.npy'
        data_entropy = np.load(data_path_entropy)
        data_partial_contr = np.load(data_path_partial_contr)
        data_std_C = np.load(data_path_std_c)
        data = [data_entropy, data_partial_contr, data_std_C]
    else:
        data = {}
        for info in plot_info:
            data_path = f'../{dataset}/info_data/{info}/frames{save_type}/{filename.split("/")[-1]}.npy'
            data[info] = np.load(data_path)
    return data


def split_video_and_record_events(array, start_ts, end_ts, stack_events):
    '''
        Captures and saves frames as a video starting from start_ts and ending at end_ts
    '''
    pass




def split_video_and_record_frames(array, output_file, start_index, end_index, fps=4):
    '''
        Captures and saves frames as a video starting from start_ts and ending at end_ts
    '''
    sliced_array = array[start_index:end_index]
    height, width = array[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', 'DIVX', etc.
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)
    print(out)
    for frame in sliced_array:
        # Write the frame into the video file
        out.write(frame[::-1])
    
    # Release the VideoWriter object
    out.release()
    # print(f"Video saved as {output_file}")


def video_frames(filename, plot_info=None):
    '''
        Creates a video from the hdf5 file
    '''
    dataset = 'info_DDD20' if 'DDD20' in filename else 'info_DSEC'
    save_type = '/'+filename.split('/')[-2] if 'DDD20' in filename else ''
    save_name = (filename.split('/')[-1]).split('.')[0]
    output_file = f'../videos/{save_type}-{save_name}.mp4' if dataset == 'info_DDD20' else f'../videos/{save_name}-no_scatter.mp4'
    frames = Frames(filename)
    print('Creating video')
    height, width = frames.frames_as_array[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', 'DIVX', etc.
    if frames.frames_as_array[0].ndim == 2:
        out = cv2.VideoWriter(output_file, fourcc, 20, (width, height), isColor=False)
    else:
        out = cv2.VideoWriter(output_file, fourcc, 20, (width, height), isColor=True)

    data = fetch_data_to_write(plot_info, dataset, save_type, filename)

    for idx, frame in tqdm(enumerate(frames.frames_as_array), total=frames.frames_as_array.shape[0]):
        if dataset == 'info_DDD20':
            # frame = write_in_frame(frame, plot_info, data, idx, rotate=True)
            out.write(frame[::-1])
        else:
            # frame = write_in_frame(frame, plot_info, data, idx, rotate=False)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()






def video_frames_with_scatter(filename, optimal_distance, plot_info=None):
    '''
        Creates a video from the hdf5 file
    '''
    dataset = 'info_DDD20' if 'DDD20' in filename else 'info_DSEC'
    save_type = '/'+filename.split('/')[-2] if 'DDD20' in filename else ''
    save_name = (filename.split('/')[-1]).split('.')[0]
    output_file = f'../videos/{save_type}-{save_name}.mp4' if dataset == 'info_DDD20' else f'../videos/{save_name}-rgb.mp4'
    frames = Frames(filename)
    print('Creating video')
    height, width = frames.frames_as_array[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', 'DIVX', etc.
    if frames.frames_as_array[0].ndim == 2:
        out = cv2.VideoWriter(output_file, fourcc, 20, (500, 700), isColor=False)
    else:
        out = cv2.VideoWriter(output_file, fourcc, 20, (500, 700), isColor=True)

    data = fetch_data_to_write(plot_info, dataset, save_type, filename)
    data = data['partial_contrast']
    distance = np.sum(np.square((data-optimal_distance)), axis=1)

    for idx, frame in tqdm(enumerate(frames.frames_as_array), total=frames.frames_as_array.shape[0]):
        if dataset == 'info_DDD20':
            # frame = write_in_frame(frame, plot_info, data, idx, rotate=True)
            fig, ax = plt.subplots(figsize=(5, 7))
            ax.imshow(frame)
            ax.axis('off')
            ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
            colors = ['red' if  d > 0.035 else 'blue' for d in distance[:idx]]
            ax2.scatter(np.linspace(0, len(distance)-1, len(distance))[:idx], distance[:idx], c=colors, marker='.', s=2,)
            ax2.set_ylabel('Distance to mean distribution')
            ax2.set_xlabel('Time')
            frame_path = f'frame_{idx}.png'
            plt.savefig(frame_path)
            plt.close()
            frame = cv2.imread(frame_path)
            out.write(frame[::-1])
        else:
            # frame = write_in_frame(frame, plot_info, data, idx, rotate=False)
            fig, ax = plt.subplots(figsize=(5, 7))
            ax.imshow(frame)
            ax.set_position([0.08, 0.4, 0.84, 0.6])
            ax.axis('off')
            ax2 = fig.add_axes([0.25, 0.2, 0.5, 0.15])
            colors = ['red' if  d > 0.035 else 'blue' for d in distance[:idx]]
            ax2.scatter(np.linspace(0, len(distance)-1, len(distance))[:idx], distance[:idx], c=colors, marker='.', s=2,)
            ax2.set_ylabel('Distance to mean distribution')
            ax2.set_xlabel('Time')
            frame_path = f'frame_{idx}.png'
            plt.savefig(frame_path)
            plt.close()
            frame = cv2.imread(frame_path)
            out.write(frame)
    for i in range(frames.frames_as_array.shape[0]):
        os.remove(f'frame_{i}.png')

    out.release()





def video_frames_with_pad(filename, optimal_distance, plot_info=None):
    '''
        Creates a video from the hdf5 file
    '''
    dataset = 'info_DDD20' if 'DDD20' in filename else 'info_DSEC'
    save_type = '/'+filename.split('/')[-2] if 'DDD20' in filename else ''
    save_name = (filename.split('/')[-1]).split('.')[0]
    output_file = f'../videos/{save_type}-{save_name}.mp4' if dataset == 'info_DDD20' else f'../videos/{save_name}-pad.mp4'
    frames = Frames(filename)
    print('Creating video')
    height, width = frames.frames_as_array[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', 'DIVX', etc.
    if frames.frames_as_array[0].ndim == 2:
        out = cv2.VideoWriter(output_file, fourcc, 20, (height+40, width+40), isColor=False)
    else:
        out = cv2.VideoWriter(output_file, fourcc, 20, (width+40, height+40), isColor=True)

    data = fetch_data_to_write(plot_info, dataset, save_type, filename)
    data = data['partial_contrast']
    distance = np.sum(np.square((data-optimal_distance)), axis=1)

    for idx, frame in tqdm(enumerate(frames.frames_as_array), total=frames.frames_as_array.shape[0]):
        image_pad = np.zeros((height+40, width+40, 3), dtype=np.uint8)
        if distance[idx] > 0.035:
            image_pad[:,:,0] = 255  
        else:
            image_pad[:,:,2] = 255
        image_pad[20:height+20, 20:width+20, :] = frame
        # plt.imshow(image_pad)
        # plt.savefig('test_ffg.png')
        # cr
        out.write(cv2.cvtColor(image_pad, cv2.COLOR_RGB2BGR))

    out.release()



def get_slices(distance, thr = 0.025, append = 30):
    condition_met = distance > thr
    window = np.ones(2 * append + 1, dtype=int)
    condition_window = np.convolve(condition_met, window, mode='same')
    valid_indices = np.where(condition_window > 5)[0]
    grouped_slices = []
    if len(valid_indices) > 0:
        temp = [valid_indices[0]]  # Start with the first index
        for i in range(1, len(valid_indices)):
            if valid_indices[i] == valid_indices[i - 1] + 1:
                temp.append(valid_indices[i])
            else:
                grouped_slices.append(temp)
                temp = [valid_indices[i]]
        grouped_slices.append(temp)  # Append the last group

    return grouped_slices
    

def video_frames_split(filename, optimal_distance, plot_info=None):
    '''
        Creates a video from the hdf5 file
    '''
    dataset = 'info_DDD20' if 'DDD20' in filename else 'info_DSEC'
    save_type = '/'+filename.split('/')[-2] if 'DDD20' in filename else ''
    save_name = (filename.split('/')[-1]).split('.')[0]
    data = fetch_data_to_write(plot_info, dataset, save_type, filename)
    data = data['partial_contrast']
    distance = np.sum(np.square((data-optimal_distance[0])/1), axis=1)
    slices = get_slices(distance)
    if slices == []:
        return
    frames = Frames(filename)
    print('Creating video')
    height, width = frames.frames_as_array[0].shape[:2]


    for idx, slice in enumerate(slices):
        output_file = f'../videos/split_red/{save_name}-{idx}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', 'DIVX', etc.
        if frames.frames_as_array[0].ndim == 2:
            out = cv2.VideoWriter(output_file, fourcc, 20, (height+40, width+40), isColor=False)
        else:
            out = cv2.VideoWriter(output_file, fourcc, 20, (width+40, height+40), isColor=True)
        for i in slice:
            frame = frames.frames_as_array[i]
            image_pad = np.zeros((height+40, width+40, 3), dtype=np.uint8)
            if distance[i] > 0.025:
                image_pad[:,:,0] = 255  
            else:
                image_pad[:,:,2] = 255
            image_pad[20:height+20, 20:width+20, :] = frame
            # plt.imshow(image_pad)
            # plt.savefig('test_ffg.png')
            # cr
            out.write(cv2.cvtColor(image_pad, cv2.COLOR_RGB2BGR))
        out.release()




def video_events(filename):
    pass




def partial_contrast_plot(name, optimal_disrt):
    '''
        If the info data for partial contrast are the 5 bins then this creates the distance to the optimal distr
    '''
    pass

