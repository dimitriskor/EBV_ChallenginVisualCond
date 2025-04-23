from ultralytics import YOLO
import os
import torch
from ultralytics.engine.results import Boxes
import pickle
import argparse

names = os.listdir('e2f_gray_trained/m_True_True4/weights')
weights = []
for n in names:
    weights.append(n) if 'epoch' in n else n

weights.sort()

# for idx, w in enumerate(weights):
#     model = YOLO(f"e2f/val_x_True_True2/weights/{w}")
#     model.train(data='../config/dummy_train.yaml', epochs=1, device=1, imgsz=640, project='epochs_train', name=f'epoch_{idx}', lr0=0.0000001, lrf=0.0000001)
    
# for idx, w in enumerate(weights):
#     model = YOLO(f"e2f_gray_trained/x_True_True/weights/{w}")
#     model.train(data='../config/e2f_gray_trained_hold.yaml', epochs=1, device=0, imgsz=640, project='e2f_epochs_hold_tr_x', name=f'epoch_{w[5:]}', lr0=0.0000001, lrf=0.0000001)

for idx, w in enumerate(weights):
    model = YOLO(f"yolo11x")
    model.train(data='../config/e2f_gray_trained_val.yaml', epochs=1, device=0, imgsz=640, project='test_pretrained', name=f'epoch_{w[5:]}', lr0=0.0000001, lrf=0.0000001)
