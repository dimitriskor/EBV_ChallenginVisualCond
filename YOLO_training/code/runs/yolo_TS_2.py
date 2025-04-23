from ultralytics import YOLO
import cv2
import os
import torch
from ultralytics.engine.results import Boxes
import pickle
import argparse

args = argparse.ArgumentParser()
args.add_argument('-c', '--config', type=str, help='Config file for dataset')
args.add_argument('-d', '--device_id', type=int, default=0, help='Id of cuda to use')
args = args.parse_args()

model = YOLO(f"glare_bins_80_clip_8_TS_1/train2/weights/last.pt")

# Train the model with MPS
results = model.train(data=f"../config/{args.config}.yaml", epochs=100, imgsz=640, device=[args.device_id], project=args.config,
                      hsv_h = 0.0,
                      hsv_s = 0,
                      hsv_v = 0,
                      degrees = 1,
                      shear = 1,
                      mosaic = 0.0,
                      scale = 0.0,
                    #   mixup = 0.3,
                    #   copy_paste = 0.3
                      )
