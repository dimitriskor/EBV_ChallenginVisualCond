from ultralytics import YOLO
import norse
import torch.nn as nn
import torch
import utils.YOLOTrain as yt
import utils.YOLOModel as ym
from tqdm import tqdm

trainset = ['interlaken_00_a',
            'interlaken_00_b',
            'interlaken_00_c',
            'thun_00_a',
            'thun_01_a',
            'thun_01_b',
            'interlaken_01_a',
            'zurich_city_03_a',
            'zurich_city_04_a',
            'zurich_city_04_b',
            'zurich_city_04_c',
            'zurich_city_04_d',
            'zurich_city_04_e',
            'zurich_city_11_a',
            'zurich_city_11_b',
            'zurich_city_11_c',
            'zurich_city_13_a',
            'zurich_city_13_b',
            'zurich_city_15_a',
            'zurich_city_00_a',
            'zurich_city_00_b',
            'zurich_city_01_a',
            'zurich_city_01_b',
            'zurich_city_01_c',
            'zurich_city_01_d',
            'zurich_city_01_e',
            'interlaken_00_d',
            'interlaken_00_e',
            'interlaken_00_f',
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
            'zurich_city_01_f',
            'zurich_city_02_a',
            'zurich_city_02_b',
            'zurich_city_02_c',
            'zurich_city_02_d',
            'zurich_city_02_e',
            'zurich_city_05_a',
            'zurich_city_05_b',
            'zurich_city_06_a',
            'zurich_city_07_a',
            'zurich_city_08_a',
]

"""trainset = ['interlaken_00_a',
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
"""
trainset = yt.HDF5Dataset_YOLO(trainset, 10, 50)
# testset = yt.HDF5Dataset_YOLO(testset, 20, 50)
batch_size=4
num_workers=16
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# model = ym.YOLOXTiny(num_classes=3)
# model = ym.YOLOSimple(num_classes=3)
# model = norse.torch.Lift(model)
grid_size=12

model = ym.YOLOBase(grid_size=grid_size, num_classes=9, bbox_per_cell=3)
model = norse.torch.Lift(model)
model = model.to('cuda:1')
# loss_fn = yt.YOLOLoss()
loss_fn = yt.YoloLossBase(grid_size=grid_size, bbox_per_cell=3, num_classes=9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
for epoch in range(200):
    for data, target, flag in tqdm(trainloader):
        if torch.sum(flag):
            continue
        data, target = data.to('cuda:1'), target.to('cuda:1')
        if len(target) != batch_size:
            continue
        target = yt.process_labels(target, batch_size, grid_size, 9, bbox_per_cell=3)
        data = data.permute(1, 0, 4, 2, 3).float()
        output = model(data)
        loss = loss_fn(output[-1], target)

        # print(loss)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        # del target
    torch.save(model.state_dict(), 'yolov1_model_v2_on_Glare_clean.pt')
    print(torch.mean(torch.tensor(losses)), epoch)
        


        



# data, target =trainset.__getitem__(150)
# data, target = data.to('cuda:1'), target.to('cuda:1')
# data = data.unsqueeze(0)
# data = data.permute(1, 0, 4, 2, 3).float()
# data = data[0]
# output = model(data)
# # output = yt.temporal_forward_function(model, data)
# output = model(data[0])
# loss = loss_fn(output, target)
# print(loss)
# loss.backward()

# optimizer.step()



# for data, target in trainloader:
#     data, target = data.to('cuda:1'), target.to('cuda:1')
#     data = data.permute(1, 0, 4, 2, 3).float()
#     target = yt.preprocess_targets(target, [(20,20),(40,40),(80,80)], 3)
#     for data_t in data:
#         cls_outs, reg_outs = model(data_t)

#     predictions = []
#     for cls_out, reg_out in zip(cls_outs, reg_outs):
#         prediction = torch.cat([reg_out, cls_out], dim=1)
#         predictions.append(prediction)
#     predictions = torch.cat([p.flatten(start_dim=2).permute(0, 2, 1) for p in predictions], dim=1)
#     loss = loss_fn(predictions, target)

    # ym.reset_model_states(model)
    # loss = loss_fn(predictions, target)
    # print(loss)
    # loss.backward()

    # optimizer.step()
    # optimizer.zero_grad()

    


# for data, target in trainloader:
#     data, target = data.to('cuda:1'), target.to('cuda:1')
#     data = data.permute(1, 0, 4, 2, 3).float()
#     print(target.shape)
#     output = model(data)
#     yt.preprocess_targets(target, [20,20], 3)  
#     print(target)
#     loss = loss_fn(output, target)

#     print(loss)
#     loss.backward()

#     optimizer.step()
#     optimizer.zero_grad()





