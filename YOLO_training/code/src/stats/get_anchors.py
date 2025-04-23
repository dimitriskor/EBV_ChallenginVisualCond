import torch
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

obj_prefix='/mnt/raid0a/Dimitris/DSEC/object_detection_yolo/'

heights = []
widhts = []
for file in os.listdir(obj_prefix):
    if '.npy' in file:
        continue
    with open(obj_prefix+file, 'rb') as f:
        targets = pickle.load(f)
        for target in targets:
            try:
                target_cx = (target[:, 0] + target[:, 2])/2
                target_w = torch.abs((target[:, 0] - target[:, 2]))
                target_cy = (target[:, 1] + target[:, 3])/2
                target_h = torch.abs(target[:, 1] - target[:, 3])
                for h in target_h:
                    heights.append(h.item())
                for w in target_w:
                    widhts.append(h.item())
            except Exception as e:
                # print(e, target)
                pass

inp = list(zip(heights, widhts))
kmeans1 = KMeans(n_clusters=6, random_state=42).fit(inp)
kmeans2 = KMeans(n_clusters=6, random_state=17).fit(inp)
kmeans3 = KMeans(n_clusters=6, random_state=0).fit(inp)
kmeans4 = KMeans(n_clusters=6, random_state=98).fit(inp)
kmeans5 = KMeans(n_clusters=6, random_state=6).fit(inp)
print(kmeans1.cluster_centers_)
print(kmeans2.cluster_centers_)
print(kmeans3.cluster_centers_)
plt.scatter(heights, widhts, marker='.', s=0.1)
def print_ratio(km):
    z1, z2, z3, z4, z5, z6 = km.cluster_centers_
    print()
    print(z1[0]/z1[1])
    print(z2[0]/z2[1])
    print(z3[0]/z3[1])
    print(z4[0]/z4[1])
    print(z5[0]/z5[1])
    print(z6[0]/z6[1])
    print()
# plt.ylim(0,200)
# plt.xlim(0,200)
print_ratio(kmeans1)
print_ratio(kmeans2)
print_ratio(kmeans3)
print_ratio(kmeans4)
print_ratio(kmeans5)
plt.savefig('k_means.png')
