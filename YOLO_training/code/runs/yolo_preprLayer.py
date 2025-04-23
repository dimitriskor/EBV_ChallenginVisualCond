from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO
# from ultralytics.utils.metrics import fitness
from ultralytics.utils.torch_utils import de_parallel
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.loader import ReprLoader
# import albumentations as A
import numpy as np
import cv2



# **Parse Arguments**
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='Config file for dataset')
parser.add_argument('-d', '--device_id', type=int, default=0, help='Id of CUDA to use')
parser.add_argument('-s', '--model_size', type=str, default='m', help='Model size. Options are n, s, m, l, x')
args = parser.parse_args()


def proc(labels):
    dict = {'cls' : [],
            'bboxes': [],
            'batch_idx': []}
    for idx, b in enumerate(labels):
        for l in b:
            # print(l)
            dict['cls'].append(torch.tensor([0.]))
            dict['bboxes'].append(torch.tensor([l[1], l[2], l[3], l[4]]))
            dict['batch_idx'].append(idx)
    if dict['cls']:  
        dict['cls'] = torch.stack(dict['cls'])
    else:
        dict['cls'] = dict['cls'] = torch.empty(0, 1) 
    if dict['bboxes']:  
        dict['bboxes'] = torch.stack(dict['bboxes'])
    else:
        dict['bboxes'] = torch.empty(0, 4)

    if dict['batch_idx']:  
        dict['batch_idx'] = torch.tensor(dict['batch_idx'])
    else:
        dict['batch_idx'] = torch.empty(0, dtype=torch.long)



    # if dict['cls'].numel() == 0:
    #     dict['cls'] = torch.tensor([[]])
    #     print('here')
    # if dict['bboxes'].numel() == 0:
    #     print('here')
    #     dict['bboxes'] = torch.tensor([[]])
    # if dict['batch_idx'].numel() == 0:
    #     print('here')
    #     dict['batch_idx'] = torch.tensor([])
    # print(dict)
    return dict


# **Set Device**
device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

# **Load YOLO Model**
yolo_model = YOLO(f"../config/{args.config}{args.model_size}.yaml").to(device)
criterion = yolo_model.loss


# **Extract Hyperparameters from YOLO**
# hyp = yolo_model.overrides
hyp = {'batch': 16,
       'lr0': 0.01,
       'lrf': 0.01,
       'weight_decay': 0.0005,
       'epochs': 40
       }


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized target tensors.
    """
    images, targets, glare_flag, input_flag = zip(*batch)  # Separate images and targets

    # Stack images (assumes images are tensors of the same shape)
    try:
        images = torch.tensor(np.array(images))
    except:
        # print(len(images))
        input_flag = [0]
    # Targets remain as a list of tensors (each tensor contains ground truth for one image)
    return images, targets, glare_flag, input_flag

# **Attach Preprocessing Module Before YOLO**
class CustomYOLOPipeline(nn.Module):
    def __init__(self, yolo):
        super(CustomYOLOPipeline, self).__init__()
        # self.preprocess_layer = TrainablePreprocessing(in_channels=10, out_channels=3)
        self.yolo = de_parallel(yolo.model)  # Remove potential DataParallel wrapper
        self.stride = getattr(self.yolo, 'stride', 32)  # Default to 32 if not found
        self.names = getattr(self.yolo, 'names', {'car': 2})

    def forward(self, x, *args, **kwargs):
        # x = self.preprocess_layer(x)  # Apply trainable preprocessing
        return self.yolo(x, *args, **kwargs)  # Pass to YOLO
    def fuse(self, verbose=False):
        """Required by Ultralytics YOLO during validation/inference."""
        if verbose:
            print("Fusion is not needed for CustomYOLOPipeline. Returning self.")
        return self  # Return self as a no-op (no fusing needed)

# **Replace YOLO's Backbone**
yolo_model.model = CustomYOLOPipeline(yolo_model)


# **Create Train and Validation DataLoader**
train_dataset = ReprLoader([], ['night'], True, 4, 'seq', (5, 10, 20, 40, 80), classes={'car' : 2}, augmentations=['jitter', 'noise', 'scale_trans'])
val_dataset = ReprLoader([], ['night'], False, 4, 'seq', (5, 10, 20, 40, 80), classes={'car' : 2})
# print(val_dataset.__getitem__(1))
# train_dataset.__getitem__(2)
train_loader = DataLoader(train_dataset, batch_size=hyp['batch'], shuffle=True, collate_fn=collate_fn, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=hyp['batch'], shuffle=False, collate_fn=collate_fn, num_workers=16)

# **Create Optimizer and Scheduler**
optimizer = optim.Adam(yolo_model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyp['lrf'], epochs=hyp['epochs'], steps_per_epoch=len(train_loader))

# **Define Loss Function (Same as YOLO)**

# **Training Loop**
epochs = hyp['epochs']
for epoch in range(epochs):
    yolo_model.model.train()
    total_loss = 0

    for batch_idx, (inputs, labels, glare_flag, input_flag) in enumerate(train_loader):
        # break
        if sum(glare_flag) or (sum(input_flag)-hyp['batch']):
            # print(sum(glare_flag))
            # print(not (sum(input_flag)-hyp['batch']))
            # print(input_flag)
            # print('skipped')
            continue
        inputs, labels = inputs.to(device), [t.to(device) for t in labels]  
        optimizer.zero_grad()
        
        # Forward pass (Preprocessing + YOLO)
        inputs = inputs.permute(0,3,1,2).float()
        inputs = F.pad(inputs, (0, 0, 80, 80))

        preds = yolo_model.yolo(inputs)

        # Compute YOLO loss
        labels = proc(labels)
        # print(labels)
        loss, loss_items = criterion(labels, preds)
        loss.backward()
        optimizer.step()

        # Track loss components
        box_loss, class_loss, dfl_loss = loss_items[:3]
        total_loss += loss.item()

        # **Print Loss in YOLO Format**
        print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | "
              f"box_loss: {box_loss:.4f}, class_loss: {class_loss:.4f}, dfl_loss: {dfl_loss:.4f}", end='\r')

        # **Step Scheduler**
        scheduler.step()
    print()
    run_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    torch.save(yolo_model.yolo.state_dict(), f'../../models/model_{args.model_size}_5ch_cl3_{run_name}.pt')
    # **Validation Loop**
    yolo_model.eval()
    with torch.no_grad():
        total_precision, total_recall, total_map50, total_map50_95 = 0, 0, 0, 0
        box_loss, class_loss, dfl_loss = 0, 0, 0
        for batch_idx, (inputs, labels, glare_flag, input_flag) in enumerate(val_loader):
            if sum(glare_flag) or (sum(input_flag)-hyp['batch']):
                # print(sum(glare_flag))
                # print(not (sum(input_flag)-hyp['batch']))
                # print(input_flag)
                continue            
            inputs, labels = inputs.to(device), [t.to(device) for t in labels]  
            inputs = inputs.permute(0,3,1,2).float()
            inputs = F.pad(inputs, (0, 0, 80, 80))
            preds = yolo_model.yolo(inputs)
            # Compute loss for validation
            labels = proc(labels)
            try:
                loss, loss_items = criterion(labels, preds)
                box_loss += loss_items[0]
                class_loss += loss_items[1]
                dfl_loss += loss_items[2]
            except:
                pass
            # # **Compute YOLO Metrics**
            # metrics = yolo_model.metrics  # Extract YOLO-style metrics
            # precision, recall, map50, map50_95 = metrics['metrics/precision(B)'], metrics['metrics/recall(B)'], metrics['metrics/mAP50(B)'], metrics['metrics/mAP50-95(B)']
            # total_precision += precision
            # total_recall += recall
            # total_map50 += map50
            # total_map50_95 += map50_95

        print(f"Validation Batch [{batch_idx+1}/{len(val_loader)}] | "
              f"box_loss: {box_loss/len(val_loader):.4f}, class_loss: {class_loss/len(val_loader):.4f}, dfl_loss: {dfl_loss/len(val_loader):.4f}")

        # # **Print Mean Metrics for the Epoch**
        # print(f"Validation Results | Precision: {total_precision/len(val_loader):.4f}, "
        #       f"Recall: {total_recall/len(val_loader):.4f}, "
        #       f"mAP@50: {total_map50/len(val_loader):.4f}, "
        #       f"mAP@50-95: {total_map50_95/len(val_loader):.4f}")
















# **Trainable Preprocessing Module**
# class TrainablePreprocessing(nn.Module):
#     def __init__(self, in_channels=10, out_channels=3, kernel_size=3):
#         super(TrainablePreprocessing, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x






# **Define Augmentations (Using Albumentations)**
# train_transforms = A.Compose([
#     A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0, p=1.0),
#     A.Rotate(limit=30, p=1.0),
#     A.Affine(shear=30, p=1.0),
# ])








# **Custom Dataset with Augmentations**
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, data, targets, transform=None):
#         self.data = data  # 10D input
#         self.targets = targets
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img = self.data[idx].numpy()  # Convert tensor to NumPy for Albumentations
#         img = img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

#         if self.transform:
#             augmented = self.transform(image=img)
#             img = augmented['image']

#         img = img.transpose(2, 0, 1)  # Back to (C, H, W)
#         img = torch.tensor(img, dtype=torch.float32)

#         return img, self.targets[idx]





