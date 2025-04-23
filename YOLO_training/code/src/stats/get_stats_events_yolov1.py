from scipy.special import kl_div
from copy import copy
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sort.tracker import SortTracker



def get_objects_per_frame(data, model_stats):
    for frame in data:
        model_stats['frames'] += 1
        if frame == [None]:
            continue
        for objects in frame:
            for object in objects:
                obj_name = object[-1]
                model_stats[obj_name] += 1
    return model_stats

def get_confidence_list(data, array):
    for frame in data:
        if frame == [None]:
            continue
        for object in frame:
            array.append(object[-2])

def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def get_objects_center_in_frame(frame):
    frame_classes = {0 : [],
                1 : [],
                2 : [],
        }
    for objects in frame:
        for object in objects:    
            center = (object[0], object[1])
            frame_classes[object[-1]].append(center)
    return frame_classes


def get_tracks(data):
    classes = {0 : [],
                1 : [],
                2 : [],
        }
    for frame in data:
        if frame == [None]:
            continue
        frame_classes = get_objects_center_in_frame(frame)
        for key in frame_classes.keys():
            classes[key].append(frame_classes[key])
    found_tracks = []
    for key in classes.keys():
        found_tracks_in_class = []
        alive_tracks = []
        all_objects = classes[key]
        for f_id, obj_in_frame in enumerate(all_objects):                                       # For every frame
            for obj in obj_in_frame:                                                            # For every object in the frame
                center = obj                                                                    # get the center
                if alive_tracks == []:                                                          # and if there are no alive tracks
                    alive_tracks.append([[center, f_id]])                                         # add that center as a new track
                    continue                                                                    # and go to the next object
                distances = []                                                                  # otherwise find the distance of the center of the object
                for track in alive_tracks:                                      
                    distances.append(calculate_distance(track[-1][0], center))                  # with the center of the alive tracks (their last element)
                min_dist = np.min(np.array(distances))                                          # get the minimum distance and
                t_id = np.argmin(np.array(distances))                                           # get the minimum distance and
                if min_dist < 40:                                                               # if it's less than 20 pixels (Eucl distance) then add it to the track
                    alive_tracks[t_id].append([center, f_id])
            to_remove = []  
            for t_id, track in enumerate(alive_tracks):                                         # before checking the new frame, move tracks that are interrupted to the found_tracks_in_class
                if f_id - track[-1][1] >= 3:
                    found_tracks_in_class.append(copy(track))
                    to_remove.append(t_id)
            for t_id in sorted(to_remove, reverse=True):                                        # Delete tracks in reverse order to avoid index shifting issues
                del alive_tracks[t_id]
        to_remove = []
        for t_id, track in enumerate(found_tracks_in_class):
            if len(track) < 6:
                to_remove.append(t_id)
        for t_id in sorted(to_remove, reverse=True):
            del found_tracks_in_class[t_id]
        
        found_tracks += found_tracks_in_class
    return found_tracks


def plot_objects_per_frame(data, cl_gl):
    model_stats_dec = {'frames' : 0,
                'two-wheeler' : 0,
                'person' : 0,
                'car' : 0,
        }
    for idx, k in enumerate(model_stats_dec.keys()):
        model_stats_dec[k] = data[list(data.keys())[idx]]
    columns = list(model_stats_dec.keys())[1:]
    values = list(model_stats_dec.values())[1:]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create a table and add it to the plot
    table_data = [[key, f'{value/data["frames"]*100:.3f}'] for key, value in zip(columns, values)]
    table = ax.table(cellText=table_data, colLabels=["Object", "Count (%)"], cellLoc='center', loc='center')

    # Adjust table formatting
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale table size for readability
    plt.savefig(f'../statistics/events-obc_count-{cl_gl}.png')
    plt.close()


def plot_conf_hist(conf, cl_gl):
    plt.figure()
    plt.hist(conf, bins = 50, density=True)
    plt.savefig(f'../statistics/events-conf-{cl_gl}.png')
    plt.close()
    return np.histogram(conf, bins=100, density=True)[0], np.mean(np.array(conf))


def track_lenght_hist(tracks, cl_gl):
    l = []
    for tr in tracks:
        l.append(len(tr))
    plt.figure()
    plt.hist(l, bins = 100, range=(1, 200), density=True)
    plt.savefig(f'../statistics/events-track_hist-{cl_gl}.png')
    plt.close()
    return np.histogram(l, bins=100, density=True)[0], np.mean(np.array(l))



def run(cl_gl):
    
    prefix = f'../statistics/events/{cl_gl}/'
    files = list(os.listdir(prefix))
    filenames = []
    for f in files:
        if f[-8:] == 'list.pkl':
            filenames.append(f)

    model_stats = {'frames' : 0,
                    0 : 0,
                    1 : 0,
                    2 : 0,
            }

    confidences = []
    found_tracks = []
    for filename in filenames:
        with open(prefix+filename, 'rb') as f:
            data = pickle.load(f) 
        model_stats = get_objects_per_frame(data, model_stats)
        get_confidence_list(data, confidences)
        found_tracks += get_tracks(data)




    plot_objects_per_frame(model_stats, cl_gl)
    hist_conf, mean_conf = plot_conf_hist(confidences, cl_gl)
    hist, mean = track_lenght_hist(found_tracks, cl_gl)
    print(f'Tracks found in {cl_gl}:',len(found_tracks))
    print(f'Mean track length in events {cl_gl}:', mean)
    print(f'Mean confidence events {cl_gl}:', mean_conf)
    print()
    print(model_stats)

    return hist + 1e-10, hist_conf + 1e-10, found_tracks, confidences









def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Each box is [top_x, top_y, bot_x, bot_y].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_tracks(frames_objects, iou_threshold=0.5):
    """
    Calculate tracks for objects across frames.
    
    Args:
    - frames_objects: List of frames, where each frame is a list of objects.
    - iou_threshold: IoU threshold to link objects between frames.
    
    Returns:
    - List of tracks, where each track is a list of objects from consecutive frames.
    """
    tracks = []  # Initialize empty list of tracks
    next_track_id = 0  # Unique ID for new tracks
    
    # List to store the current active tracks
    active_tracks = []

    for frame_idx, frame in enumerate(frames_objects):
        if frame_idx == 0:
            # Initialize tracks with the objects from the first frame
            for obj in frame:
                active_tracks.append({'id': next_track_id, 'object': obj, 'track': [obj]})
                next_track_id += 1
        else:
            # Match objects in the current frame to active tracks
            used_indices = set()  # To avoid reusing objects in the same frame
            for track in active_tracks:
                best_match = None
                best_iou = 0
                for idx, obj in enumerate(frame):
                    if idx in used_indices:
                        continue
                    iou = calculate_iou(track['object'][:4], obj[:4])  # Compare bounding boxes
                    if iou > best_iou and iou >= iou_threshold:
                        best_match = (idx, obj)
                        best_iou = iou
                
                # Update track if a match was found
                if best_match:
                    idx, obj = best_match
                    used_indices.add(idx)
                    track['object'] = obj
                    track['track'].append(obj)
            
            # Add new tracks for unmatched objects
            for idx, obj in enumerate(frame):
                if idx not in used_indices:
                    active_tracks.append({'id': next_track_id, 'object': obj, 'track': [obj]})
                    next_track_id += 1
    
    # Extract tracks from the active tracks
    return [track['track'] for track in active_tracks]








data = torch.load('output_yolov1_model_v2_on_Glare_clean.pt')
conf = []
get_confidence_list(data[:10], conf)
conf = torch.tensor(conf)
print('Mean confidence', torch.mean(conf))
total_obj = []
for obj in data:
    total_obj.append(obj)
print('Found objects', len(obj)/len(data))

tracker = SortTracker(max_age=10)
tracks = []
for image in data:
   try:
       image[:] = image[:].squeeze(-1) 
   except:
       continue
   online_targets = tracker.update(image, 0)
   tracks.append(online_targets)

from collections import defaultdict

# Create a dictionary to store track lengths
track_lengths = defaultdict(int)  # {track_id: length}

# Aggregate track lengths
for frame_tracks in tracks:  # Loop through each frame's tracks
    for track in frame_tracks:
        track_id = track[-1]  # Assuming the track ID is the last element in each track
        track_lengths[track_id] += 1  # Increment the length for this track ID

# Extract lengths for the histogram
lengths = list(track_lengths.values())

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=range(5, 100), edgecolor='black', align='left')
plt.title('Histogram of Track Lengths')
plt.xlabel('Track Length')
plt.ylabel('Frequency')
# plt.xticks(range(1, max(lengths) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('hist_tracker.png')

track_lengths = defaultdict(int)  # {track_id: length}

# Aggregate track lengths
for frame_tracks in tracks:  # Loop through each frame's tracks
    for track in frame_tracks:
        track_id = track[-1]  # Assuming the track ID is the last element in each track
        track_lengths[track_id] += 1  # Increment the length for this track ID

# Calculate statistics
num_tracks = len(track_lengths)  # Total number of unique tracks
mean_length = sum(track_lengths.values()) / num_tracks if num_tracks > 0 else 0

print(f"Number of tracks: {num_tracks}")
print(f"Mean length of tracks: {mean_length}")

# h_clear, h_clear_conf, tr_cl, conf_cl = run(cl_gl = 'clear')
# h_glare, h_glare_conf, tr_gl, conf_gl = run(cl_gl = 'glare')

# plt.figure()
# plt.hist([len(t) for t in tr_cl], alpha=0.5, range = (1,200), density=True, bins = 100, label='Track length clear', color='b')
# plt.hist([len(t) for t in tr_gl], alpha=0.5, range = (1,200), density=True, bins = 100, label='Track length flagged', color='r')
# plt.legend()
# plt.savefig('../statistics/events-track.png')
# plt.close()

# plt.figure()
# plt.hist(conf_cl, alpha=0.5, density=True, bins = 50, label='Confidence clear', color='b')
# plt.hist(conf_gl, alpha=0.5, density=True, bins = 50, label='Confidence flagged', color='r')
# plt.legend()
# plt.savefig('../statistics/events-conf.png')
# plt.close()

# kl_divergence = np.sum(kl_div(h_clear, h_glare))
# kl_divergence_conf = np.sum(kl_div(h_clear_conf, h_glare_conf))
# print("KL Divergence of track histogram:", kl_divergence)
# print("KL Divergence of confidence histogram:", kl_divergence_conf)
