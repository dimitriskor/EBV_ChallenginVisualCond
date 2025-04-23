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
args.add_argument('-s', '--size', type=str, help='Model size (n, s, m, l ,x)')
# args.add_argument('-t', '--train_type', type=str, help='Set for validation (train, val)')
args.add_argument('-a', '--augment', action='store_true', help='Augment dataset')
args.add_argument('-p', '--pretrained', action='store_true', help='Use pretrained model')

args = args.parse_args()

model = YOLO(f"../config/yolo11{args.size}.yaml") if not args.pretrained else YOLO(f"yolo11{args.size}")  # load a pretrained model (recommended for training)



hsv_h = 0.0
hsv_s = 0
hsv_v = 0
degrees = 1
shear = 1
mosaic = 0.0
scale = 0.0
mixup = 0.0
copy_paste = 0.0
erasing = 0.1
translate = 0.1
crop_fraction = 0.0

if args.augment:
  hsv_h = 0.02
  hsv_v = 0.2
  hsv_s = 0.25
  degrees = 10
  translate = 0.25
  scale = 0.25
  shear = 5
  mosaic = 0.0
  mixup = 0.1
  copy_paste = 0.25
  erasing = 0.4
  crop_fraction = 0.2


# Train the model with MPS
results = model.train(data=f"../config/{args.config}.yaml", epochs=40, imgsz=640, device=[args.device_id], project=args.config, name=f'{args.size}_{str(args.augment)}_{str(args.pretrained)}', lr0=0.0001, lrf=0.00001, optimizer='Adam',
                      hsv_h = hsv_h,
                      hsv_s = hsv_s,
                      hsv_v = hsv_v,
                      degrees = degrees,
                      translate = translate,
                      scale = scale,
                      shear = shear,
                      mosaic = mosaic,
                      mixup = mixup,
                      copy_paste = copy_paste,
                      erasing = erasing,
                      crop_fraction = crop_fraction,
                      save_period = 2,
                      )
