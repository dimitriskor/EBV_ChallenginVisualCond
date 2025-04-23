from ultralytics import YOLO
import norse
import torch.nn as nn
import torch
import utils.YOLOTrain as yt
import utils.YOLOModel as ym
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import utils

trainset = ['interlaken_00_a',
            # 'interlaken_00_b',
            # 'interlaken_00_c',
            # 'interlaken_00_d',
            # 'interlaken_00_e',
            # 'interlaken_00_f',
            # 'thun_00_a',
            'thun_01_a',
            # 'thun_01_b',
            'zurich_city_03_a',
            'zurich_city_04_a',
            'zurich_city_04_b',
            # 'zurich_city_04_c',
            # 'zurich_city_04_d',
            # 'zurich_city_04_e',
            # 'zurich_city_05_a',
            # 'zurich_city_05_b',
            # 'zurich_city_06_a',
            # 'zurich_city_07_a',
            # 'zurich_city_08_a',
            # 'zurich_city_11_a',
            # 'zurich_city_11_b',
            # 'zurich_city_11_c',
            # 'zurich_city_13_a',
            # 'zurich_city_13_b',
            # 'zurich_city_15_a',
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

trainset = yt.HDF5Dataset_YOLO(trainset, 10, 50)
testset = yt.HDF5Dataset_YOLO(testset, 20, 50)
batch_size=1
num_workers=4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



grid_size=[(5, 5), (10,10), (20,20)]

model = ym.YOLOXTiny(num_classes=9, num_anchors=5)
model = norse.torch.Lift(model)
model.load_state_dict(torch.load('yolovX_model.pt', weights_only=True))
# model = model.to('cuda:1')
model.eval()
anchors = torch.stack([yt.generate_random_anchors(5, 0.02, 0.4, 'cpu') for _ in range(3)])

losses = []
with torch.no_grad():
    for epoch in range(50):
        for data, target in tqdm(trainloader):
            # data, target = data.to('cuda:1'), target.to('cuda:1')
            # for i in range(seq_length):
            #     plt.imshow(20*data[0][i].cpu())
            #     plt.savefig(f'image_{i}.png')
            print(data.shape, 'before permute')
            print(target)
            # utils.plot_bounding_boxes(target[0].cpu(), data[0][-1].cpu())
            if len(target) != batch_size:
                continue
            target = yt.process_labels_yolox(target, batch_size, grid_size, 9, anchor_boxes_list=anchors)
            # plt.imshow((20*data[-1][0]).clip(0, 255).to('cpu'))
            # plt.savefig('data_plt.png')
            data = data.permute(1, 0, 4, 2, 3).float()
            utils.plot_ground_truth_bboxes_yoloX(data[-1][0], target[0], grid_size, )
            output = model(data)
            print(output.shape)
            print(data.shape, 'after permute')
            boxes, scores, classes = utils.non_max_suppression_yoloX(output[-1][0], grid_size)
            utils.plot_predictions_yoloX(data[-1][0], boxes, scores, classes)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(frame/255)
            # plt.savefig('data_plt_boxes.png')
            # plt.axis('off')
            # plt.show()
            import time
            time.sleep(4)

        