import numpy as np
import utils.Events as Events
import os
import matplotlib.pyplot as plt
from utils.Events import Events
from utils.Frames import Frames
import argparse
import utils.entropy as entropy

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
args = parser.parse_args()

prefix = args.path

if 'DDD20' in prefix:
    names_clear = os.listdir(args.path+'clear')
    names_glare = os.listdir(args.path+'glare')
    names_night = os.listdir(args.path+'night')
else:
    names = os.listdir(prefix)


def clear_names(names):
    cleared = []
    for n in names:
        if 'exported' in n:
            cleared.append(n)
    return cleared



def calculate_entropy_events(names, vis_type, info_type, entropy_fn):
    names = clear_names(names) if 'DDD20' in prefix else names
    dataset = 'info_DDD20' if 'DDD20' in prefix else 'info_DSEC'
    for file in names:
        print(f'Calculating event entropy in file {file} of type {vis_type}')
        filename = prefix + vis_type + '/' + file                                                   # Used when working on DDD20
        filename = prefix + file
        events = Events(filename, stack_events=250)
        entropy = events.calculate_entropy(levels=256, entropy_fn=entropy_fn)
        np.save(f'../{dataset}/info_data/{info_type}/events/{vis_type}/{file}.npy', entropy)
        plt.figure()
        plt.plot(entropy)
        plt.title(f'File: {file} - Type: {vis_type}')
        plt.savefig(f'../{dataset}/info_plots/{info_type}/events/{vis_type}/{file}.png')
        plt.close()
        print()


def calculate_entropy_frames(names, vis_type, info_type, entropy_fn, ranges=None):
    names = clear_names(names) if 'DDD20' in prefix else names
    dataset = 'info_DDD20' if 'DDD20' in prefix else 'info_DSEC'
    for file in names:
        print(f'Calculating event entropy in file {file} of type {vis_type}')
        # filename = prefix + vis_type + '/' + file                                                   # Used when working on DDD20
        filename = prefix + file
        print(filename)
        frames = Frames(filename)
        entropy = frames.calculate_entropy(levels=256, entropy_fn=entropy_fn, ranges=ranges)
        np.save(f'../{dataset}/info_data/{info_type}/frames/{vis_type}/{file}.npy', entropy)
        plt.figure()
        # optimal_distr = [0.32-0.16, 0.42+0.16, 0.12+0.08, 0.08-0.03, 0.06-0.01]
        # entropy = np.sum(np.square(optimal_distr-entropy), axis=1)
        plt.plot(entropy)
        # plt.title(f'File: {file} - Type: {vis_type}')
        plt.title(f'File: {file}')
        plt.savefig(f'../{dataset}/info_plots/{info_type}/frames/{vis_type}/{file}.png')
        plt.close()
        print()



def calculate_entropy(names, vis_tupe, information_type = None, ranges=None):
    if information_type == None or information_type == 'entropy':
        entropy_fn = entropy.first_order_entropy
    if information_type == 'std_C':
        entropy_fn = entropy.std_C
    if information_type == 'window_std':
        entropy_fn = entropy.window_std
    if information_type == 'C_lin':
        entropy_fn = entropy.C_lin
    if information_type == 'C_wei':
        entropy_fn = entropy.C_wei
    if information_type == 'partial_contrast':
        entropy_fn = entropy.partial_contrast

    calculate_entropy_frames(names, vis_tupe, information_type, entropy_fn, ranges)
    # calculate_entropy_events(names, vis_tupe, information_type, entropy_fn)


# calculate_entropy(names_clear, 'clear', 'partial_contrast')
# calculate_entropy(names_glare, 'glare', 'partial_contrast')
# calculate_entropy(names_night, 'night', 'partial_contrast')
d = 1/255
# ranges = [(0, 1*d), (1*d, 2*d), (2*d, 4*d), (4*d, 7*d), (7*d, 12*d), (12*d, 18*d), (18*d, 26*d), (26*d, 36*d), (36*d, 48*d), (48*d, 65*d),  (65*d, 80*d),  (80*d, 98*d),  (98*d, 120*d),  (120*d, 150*d),  (150*d, 190*d), (190*d, 1)]
ranges = [0, 1*d, 2*d, 4*d, 7*d, 12*d, 18*d, 26*d, 36*d, 48*d, 65*d, 80*d, 98*d, 120*d, 150*d, 190*d, 1]

calculate_entropy(names, '', 'partial_contrast', ranges = ranges)

# calculate_entropy(names_clear, 'clear', 'C_lin')
# calculate_entropy(names_glare, 'glare', 'C_lin')
# calculate_entropy(names_night, 'night', 'C_lin')

# calculate_entropy(names_clear, 'clear', 'entropy')
# calculate_entropy(names_glare, 'glare', 'entropy')
# calculate_entropy(names_night, 'night', 'entropy')

# calculate_entropy(names_clear, 'clear', 'std_C')
# calculate_entropy(names_glare, 'glare', 'std_C')
# calculate_entropy(names_night, 'night', 'std_C')
