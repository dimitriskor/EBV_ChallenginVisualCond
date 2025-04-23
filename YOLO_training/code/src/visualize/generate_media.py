import matplotlib.pyplot as plt
import numpy as np
import utils.utils as tl

# file = np.load('../info_DSEC/info_data/partial_contrast_5delta/frames/interlaken_00_b.npy')
# plt.bar([0,1,2,3,4], file[0], width=.5)
# plt.xticks([0,1,2,3,4], labels=['insignificant', 'small', 'medium', 'large', 'extreme'])
# plt.xlabel('Neighbouring pixel difference')
# plt.ylabel('Percentage')
# plt.title('Distribution of pixel differences')
# plt.savefig('Distribution of pixel differences.png')



# file = np.load('../info_DSEC/info_data/partial_contrast_5delta/frames/interlaken_01_a.npy')
# plt.plot(file, label=['insignificant', 'small', 'medium', 'large', 'extreme'])
# plt.xlabel('Time')
# plt.ylabel('Percentage')
# plt.title('Distribution of pixel differences over time')
# plt.savefig('Distribution of pixel differences over time.png')


opt_d, _ = tl.optimal_distance('DSEC')
plt.bar([0,1,2,3,4], opt_d, width=.5)
plt.xticks([0,1,2,3,4], labels=['insignificant', 'small', 'medium', 'large', 'extreme'])
plt.xlabel('Neighbouring pixel difference')
plt.ylabel('Percentage')
plt.title('Mean distribution of pixel differences')
plt.savefig('Optimal Distribution of pixel differences.png')
