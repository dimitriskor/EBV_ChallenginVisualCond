import norse
from utils import utils
import utils.YOLOModel as ym
from utils.Events import Events
import os
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import torch
import torchvision
import utils.YOLOTrain as yt
import utils.YOLOModel as ym
from tqdm import tqdm



def identify_detections(target, output, conf_thre=0.5, nms_thre=0.7, bbox_per_cell=3):


    def preprocess(prediction, rd = False):
        grid_size = prediction.shape[0]  # Assume square grid
        frame_width, frame_height = 640, 480

        # Prepare the detections list
        detections = []

        # Iterate over all grid cells
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(bbox_per_cell):
                    # Extract prediction for one bounding box
                    bbox_start = k * 5  # Start index for bbox data (x, y, w, h, confidence)
                    bbox = prediction[i, j, bbox_start:bbox_start + 5]
                    if torch.sum(bbox[:4] == torch.tensor([0,0,0,0],  device=device)) == 4:
                        continue
                    obj_conf = bbox[4]  # Objectness confidence

                    # Extract class probabilities
                    class_probs = prediction[i, j, bbox_per_cell * 5:]
                    class_conf, class_pred = torch.max(class_probs, 0)  # Max class confidence and index
                    # Apply confidence threshold
                    if obj_conf * class_conf >= conf_thre:
                        # Get bounding box coordinates
                        top_x = (bbox[0] + j) / grid_size * frame_width  # Top-left x in image space
                        top_y = (bbox[1] + i) / grid_size * frame_height  # Top-left y in image space
                        width = bbox[2] * frame_width  # Width in image space
                        height = bbox[3] * frame_height  # Height in image space

                        # Compute (x1, y1, x2, y2)
                        x1 = top_x - width / 2
                        y1 = top_y - height / 2
                        x2 = x1 + width
                        y2 = y1 + height
                        # if rd:
                        #     x1, y1, x2, y2 = 640*x1, 480*y1, 640*x2, 480*y2
                        # Append detection: (x1, y1, x2, y2, confidence, class_id)
                        detections.append([x1, y1, x2, y2, obj_conf * class_conf, class_pred.item()])

        # Apply Non-Maximum Suppression (NMS) if detections are not empty
        if len(detections) > 0:
            detections = torch.tensor(detections)
            nms_indices = torchvision.ops.nms(
                detections[:, :4],  # Coordinates (x1, y1, x2, y2)
                detections[:, 4],   # Confidence scores
                nms_thre            # IoU threshold
            )
            detections = detections[nms_indices]

        return detections
    target_det = preprocess(target)
    pred_det = preprocess(output, True)
    # if len(target_det) > 0:
    #     print('all good')
    #     target_det = target_det.unsqueeze(-2)
    # print(target_det)
    # print(pred_det)
    def calculate_iou(box1, box2):
        """
        Compute IoU between two bounding boxes.
        box1, box2: [x_min, y_min, x_max, y_max]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        # print(x1, x2, y1, y2)

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        # print(inter_area / union_area)
        return inter_area / union_area if union_area > 0 else 0

    def ap50(predictions, ground_truths, iou_threshold=0.5):
        """
        Compute AP50 for predictions and ground truths.
        predictions: List of [x_min, y_min, x_max, y_max, confidence, class]
        ground_truths: List of [x_min, y_min, x_max, y_max, class]
        """
        predictions = sorted(predictions, key=lambda x: x[4], reverse=True)  # Sort by confidence
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        gt_used = set()
        ious = []
        for pred_idx, pred in enumerate(predictions):
            pred_box, pred_class = pred[:4], pred[5]
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in gt_used:  # Skip already matched ground truths
                    continue
                
                gt_box, gt_class = gt[:4], gt[5]
                if pred_class != gt_class:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold:
                tp[pred_idx] = 1
                gt_used.add(best_gt_idx)
                # print('here')
            else:
                fp[pred_idx] = 1
            ious.append(best_iou)

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        # print(tp_cumsum)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / len(ground_truths)

        # Compute AP as the area under the precision-recall curve
        ap = 0
        for i in range(1, len(precision)):
            ap += (recall[i] - recall[i - 1]) * precision[i]
        
        return ap, ious
         

    AP50, iou = ap50(target_det, pred_det, 0.0)
    return iou





def run_inference(model, data, folder, num_classes=8):
    '''
        Run model on inference and create video with the output.
    '''
    print(type(data))
    extra_channel = data[:, 0, :, :].unsqueeze(1)
    print(data.shape, 'before permute')
    data = torch.cat((data, extra_channel), dim=1).unsqueeze(1).float()
    print(data.shape, 'after permute')
    output = model(data)


    for idx, out in enumerate(output[10:]):
        AP50 = identify_detections(out[0], data[idx][0], num_classes)
        print(AP50)



trainset = ['interlaken_00_a',
            'interlaken_00_b',
            'interlaken_00_c',
            'interlaken_00_d',
            'interlaken_00_e',
            'interlaken_00_f',
            'thun_00_a',
            'thun_01_a',
            'thun_01_b',
            'zurich_city_03_a',
            'zurich_city_04_a',
            'zurich_city_04_b',
            'zurich_city_04_c',
            'zurich_city_04_d',
            'zurich_city_04_e',
            'zurich_city_05_a',
            'zurich_city_05_b',
            'zurich_city_06_a',
            'zurich_city_07_a',
            'zurich_city_08_a',
            'zurich_city_11_a',
            'zurich_city_11_b',
            'zurich_city_11_c',
            'zurich_city_13_a',
            'zurich_city_13_b',
            'zurich_city_15_a',
            'interlaken_01_a',
            'zurich_city_09_a',
            'zurich_city_09_b',
            'zurich_city_09_c',
            'zurich_city_09_d',
            'zurich_city_09_e',
            'zurich_city_10_a',
            'zurich_city_10_b',
            'zurich_city_12_a',
            'zurich_city_14_a',
            'zurich_city_14_b',
            'zurich_city_14_c',
            'zurich_city_00_a',
            'zurich_city_00_b',
            'zurich_city_01_a',
            'zurich_city_01_b',
            'zurich_city_01_c',
            'zurich_city_01_d',
            'zurich_city_01_e',
            'zurich_city_01_f',
            'zurich_city_02_a',
            'zurich_city_02_b',
            'zurich_city_02_c',
            'zurich_city_02_d',
            'zurich_city_02_e',
            ]
testset = ['interlaken_01_a',
            'zurich_city_09_a',
            'zurich_city_09_b',
            'zurich_city_09_c',
            'zurich_city_09_d',
            'zurich_city_09_e',
            'zurich_city_10_a',
            'zurich_city_10_b',
            'zurich_city_12_a',
            'zurich_city_14_a',
            'zurich_city_14_b',
            'zurich_city_14_c',
            'zurich_city_00_a',
            'zurich_city_00_b',
            'zurich_city_01_a',
            'zurich_city_01_b',
            'zurich_city_01_c',
            'zurich_city_01_d',
            'zurich_city_01_e',
            'zurich_city_01_f',
            'zurich_city_02_a',
            'zurich_city_02_b',
            'zurich_city_02_c',
            'zurich_city_02_d',
            'zurich_city_02_e',
            ]

device='cuda:0'
batch_size=1
grid_size=12
bin_time = 10
seq_length = 50
trainset = yt.HDF5Dataset_YOLO(trainset, bin_time, seq_length)
testset = yt.HDF5Dataset_YOLO(testset, bin_time, seq_length)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

model_name = 'yolov1_model_v2_on_Glare_clean.pt'

model = ym.YOLOBase(grid_size=grid_size, num_classes=9, bbox_per_cell=3)
model = norse.torch.Lift(model)
model.load_state_dict(torch.load(model_name, weights_only=True))
model = model.to(device)
model.eval()

losses = []
ious = []
for epoch in range(1):
    for data, target, flag in tqdm(trainloader):
        if torch.sum(flag):
            continue
        data, target = data.to(device), target.to(device)
        if len(target) != batch_size:
            continue
        target = yt.process_labels(target, batch_size, grid_size, 9, bbox_per_cell=3)
        # plt.imshow((20*data[-1][0]).clip(0, 255).to('cpu'))
        # plt.savefig('data_plt.png')
        data = data.permute(1, 0, 4, 2, 3).float()
        output = model(data)
        iou = identify_detections(output[-1][0], target[0])
        for i in iou:
            ious.append(i)
        # print(ious)
        ious_ = torch.tensor(ious)
        # print(f'ious_{model_name}')
    torch.save(ious_, f'ious_cam_over_EBV_{model_name}')

        