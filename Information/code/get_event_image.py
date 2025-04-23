import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt

a = h5py.File('../../data/DSEC/test_events/zurich_city_12_a/events/left/events.h5', 'r')

idx = 1465

start = a['ms_to_idx'][idx]
end = a['ms_to_idx'][idx+4]
events_x = a['events']['x'][start:end]
events_y = a['events']['y'][start:end]
events_p = a['events']['p'][start:end]

image = np.zeros((np.max(events_y)+1, np.max(events_x)+1))
for i in range(len(events_x)):
    image[events_y[i], events_x[i]] += events_p[i]
print(np.max(image))
# plt.imshow(image)
plt.imsave('events.png', image)