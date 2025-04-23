import os
import matplotlib.pyplot as plt
import cv2
from utils.Events import Events
from utils.Frames import Frames
from tqdm import tqdm
import numpy as np
import utils.utils as utils





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
