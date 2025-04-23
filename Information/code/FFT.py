import cv2
import numpy as np
import os
import  matplotlib.pyplot as plt
from tqdm import tqdm
recordings = os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/')
for name in recordings:
    # if name != 'zurich_city_04_f':
    if name != 'interlaken_00_b':
        continue
    rec=f'/mnt/raid0a/Dimitris/DSEC/images_recordings/{name}/images/left/rectified/'
    images = os.listdir(rec)
    images.sort()
    series = []
    for image in tqdm(images[600:1000]):
        im = cv2.imread(rec+image)
        ft = np.fft.fft2(im, axes=(0,1))
        ft = np.fft.fftshift(ft)  # Shift the zero frequency component to the center

        magnitude_spectrum = np.log(np.abs(ft) + 1)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)


        # plt.imshow(magnitude_spectrum)
        # cv2.imwrite('fvfd.png', magnitude_spectrum)
        # sd
        s_ft = np.sum(magnitude_spectrum[:400][:400]+magnitude_spectrum[-400:][-400:])
        series.append(s_ft)
    plt.plot(series)
    plt.savefig('fft.png')
    safd

