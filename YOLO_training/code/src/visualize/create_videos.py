import utils.visualize as vis
from utils.Frames import Frames
import os
import argparse
import utils.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
args = parser.parse_args()

if 'DDD20' in args.path:
    for root, dirs, files in os.walk(args.path):
        for name in files:
            if 'exported' in name:
                print('Creating video of:', root+'/'+name)
                vis.video_frames(root+'/'+name, ['partial_contrast'])

if 'DSEC' in args.path:
    names = list(os.listdir(args.path))
    for path in names:
        vis.video_frames_split(args.path+path, utils.optimal_distance('DSEC'), ['partial_contrast'])



# if 'DSEC' in args.path:
#     names = list(os.listdir(args.path))
#     for path in names:
#         vis.video_frames_with_scatter(args.path+path, utils.optimal_distance('DSEC')[0], ['partial_contrast'])