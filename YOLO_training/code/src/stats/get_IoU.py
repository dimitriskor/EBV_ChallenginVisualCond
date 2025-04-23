import torch
import numpy as np
import matplotlib.pyplot as plt


ious = torch.load('ious_cam_over_EBV_yolov1_model_v2_on_Glare_clean.pt')
# plt.hist(ious, bins=18, range=(0.1, 1))
# plt.savefig('ious.png')
ious_inv = torch.load('ious_yolov1_model_v2_on_Glare_clean.pt')
plt.hist(ious_inv, bins=18, range=(0.1, 1))
plt.xlabel('IoU')
plt.ylabel('Object count')
plt.title('Intesection over Union (IoU)')
plt.savefig('ious.png')

fp = torch.sum(ious == 0.0)/len(ious)
print(fp)
fn = torch.sum(ious_inv == 0.0)/len(ious_inv)
print(fn)

plt.close()

ious = torch.load('ious_test_yolov1_model_v2_on_Glare_clean.pt')
plt.hist(ious, bins=20)
plt.xlabel('IoU')
plt.ylabel('Object count')
plt.title('Intesection over Union (IoU) on glare input')
plt.savefig('ious_test.png')
plt.close()

# ious = torch.load('ious_test_cam_over_EBV_yolov1_model_v2_on_Glare_clean.pt')
# plt.hist(ious, bins=20)
# plt.xlabel('IoU')
# plt.ylabel('Object count')
# plt.title('Intesection over Union (IoU) on glare input')
# plt.savefig('ious_test.png')