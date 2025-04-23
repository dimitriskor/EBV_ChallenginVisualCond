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
from torchvision.ops import nms


def process_yolo_output(output, conf_threshold=0.5, nms_threshold=0.4):
    batch_bboxes = []
    
    # Iterate through each image in the batch
    for i in range(output.shape[0]):
        # Extracting x_center, y_center, width, height, and object confidence
        xywh = output[i, :4, :]  # Shape: [4, num_grid_cells]
        obj_conf = output[i, 4, :]  # Shape: [1, num_grid_cells]
        
        # Apply confidence threshold
        conf_mask = obj_conf > conf_threshold
        xywh = xywh[:, conf_mask]  # Filter out boxes with low object confidence
        obj_conf = obj_conf[conf_mask]
        
        if xywh.size(1) == 0:
            batch_bboxes.append([])
            continue
        
        # Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]
        x_center, y_center, width, height = xywh
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Prepare the bounding boxes for NMS (x1, y1, x2, y2, score)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # Shape: [num_boxes, 4]
        scores = obj_conf  # Confidence scores
        
        # Apply Non-Maximum Suppression (NMS)
        keep_indices = nms(boxes, scores, nms_threshold)
        
        # Filter boxes after NMS
        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices]
        
        # Store the results for this image
        batch_bboxes.append((final_boxes, final_scores))
    
    if batch_bboxes == [[]]:
        batch_bboxes = [([], [])]
    
    return batch_bboxes



def plot_bboxes_on_image(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image.
    
    Arguments:
    - image: the input image (H, W, 3).
    - bboxes: list of bounding boxes [(x1, y1, x2, y2), ...].
    - color: color for the bounding box (default: green).
    - thickness: thickness of the bounding box lines (default: 2).
    
    Returns:
    - Image with bounding boxes drawn.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert tensor to NumPy array if it's a tensor
    image = image.copy()
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image



def create_video_from_frames(frames, output_video_path, frame_rate=30, frame_size=(640, 480)):
    """
    Create a video from a list of frames.
    - frames: list of frames (numpy arrays).
    - output_video_path: path to save the output video.
    - frame_rate: frame rate of the video.
    - frame_size: size of the video frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    for frame in frames:
        frame = frame.astype(np.uint8)
        out.write(frame)

    out.release()



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

    return dict





# **Parse Arguments**
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='Config file for dataset')
parser.add_argument('-d', '--device_id', type=int, default=0, help='Id of CUDA to use')
parser.add_argument('-s', '--model_size', type=str, default='m', help='Model size. Options are n, s, m, l, x')
args = parser.parse_args()



# **Set Device**
device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
# **Load YOLO Model**
yolo_model = YOLO(f"../config/{args.config}{args.model_size}.yaml").to(device)
criterion = yolo_model.loss

yolo_model.model.load_state_dict(torch.load('../../models/model_n_5ch_cl3_2025-03-03_13:09:17.pt'))
predictor = yolo_model.task_map['detect']['predictor']()
print(predictor)


hyp = {'batch': 1,
       'lr0': 0.01,
       'lrf': 0.01,
       'weight_decay': 0.0005,
       'epochs': 100
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
        input_flag = [0]
    return images, targets, glare_flag, input_flag

# **Attach Preprocessing Module Before YOLO**
class CustomYOLOPipeline(nn.Module):
    def __init__(self, yolo):
        super(CustomYOLOPipeline, self).__init__()
        self.yolo = de_parallel(yolo.model)  
        self.stride = getattr(self.yolo, 'stride', 32)  
        self.names = getattr(self.yolo, 'names', {'car': 2})

    def forward(self, x, *args, **kwargs):
        return self.yolo(x, *args, **kwargs)  # Pass to YOLO
    def fuse(self, verbose=False):
        """Required by Ultralytics YOLO during validation/inference."""
        if verbose:
            print("Fusion is not needed for CustomYOLOPipeline. Returning self.")
        return self  # Return self as a no-op (no fusing needed)

val_dataset = ReprLoader([], ['night'], False, 4, 'seq', (5, 10, 20, 40, 80), classes={'car' : 2})
val_loader = DataLoader(val_dataset, batch_size=hyp['batch'], shuffle=False, collate_fn=collate_fn, num_workers=1)


# **Training Loop**
frames = []
yolo_model.eval()
with torch.no_grad():
    for batch_idx, (inputs, labels, glare_flag, input_flag) in enumerate(val_loader):
        print(batch_idx)
        if batch_idx <= 0:
            continue
        if batch_idx >= 1000:
            break
        if (sum(input_flag)-hyp['batch']):
            continue            
        try:
            inputs, labels = inputs.to(device), [t.to(device) for t in labels]  
            inputs = inputs.permute(0,3,1,2).float()
            inputs = F.pad(inputs, (0, 0, 80, 80))
            # import matplotlib.pyplot as plt
            # plt.imshow(inputs[0][:3].permute(1, 2, 0).cpu())
            # plt.savefig('asffgsdfgzsdcsz.png')
            # asfds
            preds = yolo_model.model(inputs)
            preds = predictor.inference(yolo_model.model, inputs)
            # predictor.postprocess(preds, inputs[0], inputs)
            # print(preds.shape)
            # print(preds)
            # print(yolo_model.model.model[-1]._inference(inputs))
            labels = proc(labels)
            print(preds[0].shape, len(preds[1]))
            # print(preds[1][0].shape, preds[1][1].shape, preds[1][2].shape)
            # print(preds[1][0][0, :, 0,0])
            bboxes = process_yolo_output(preds[0])
            print(bboxes)
            loss, loss_items = criterion(labels, preds)
            print(loss_items)
            final_bboxes, final_scores = bboxes[0]
            frames.append(plot_bboxes_on_image(inputs[0,:3].permute(1, 2, 0), final_bboxes))
            print('here')
        except:
            print(bboxes)
            frames.append(inputs[0,:3].permute(1, 2, 0))
            pass
create_video_from_frames(frames, 'output_video.mp4', frame_rate=20, frame_size=(640, 640))











# # **Training Loop**
# frames = []
# yolo_model.eval()
# with torch.no_grad():
#     for i in range(400,1000):
#         (inputs, labels, glare_flag, input_flag) = collate_fn(val_dataset.__getitem__(i))
#     # for batch_idx, (inputs, labels, glare_flag, input_flag) in enumerate(val_loader):
#         # if batch_idx <= 400:
#         #     continue
#         # if batch_idx >= 1000:
#         #     break
#         if (sum([input_flag])-hyp['batch']):
#             print(i)     
#             continue       
#         # try:
#         inputs = inputs.unsqueeze(0)
#         inputs, _ = inputs.to(device), labels.to(device)
#         inputs = inputs.permute(0,3,1,2).float()
#         inputs = F.pad(inputs, (0, 0, 80, 80))
#         preds = yolo_model.model(inputs)
#         # print(preds[0].shape, len(preds[1]))
#         # print(preds[1][0].shape, preds[1][1].shape, preds[1][2].shape)
#         # print(preds[1][0][0, :, 0,0])
#         bboxes = process_yolo_output(preds[0])
#         final_bboxes, final_scores = bboxes[0]
#         print(inputs[0,:3].shape, i)
#         frames.append(plot_bboxes_on_image(inputs[0,:3].permute(1, 2, 0), final_bboxes))
#         # except Exception as e:
#         #     print(i, 'except', e)
#         #     pass
#         # if batch_idx >= 600:
#         #     break
# create_video_from_frames(frames, 'output_video.mp4', frame_rate=20, frame_size=(640, 640))
