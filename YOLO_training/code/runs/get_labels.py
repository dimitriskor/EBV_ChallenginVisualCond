from matplotlib import legend
import matplotlib.pyplot as plt
import pandas as pd

filename = 'glare_bins_80_clip_8_TS_1/train2/results.csv'

df = pd.read_csv(filename)
print(df)

total_train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
total_val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']

plt.plot(total_train_loss, label='train loss')
plt.plot(total_val_loss, label='val loss')
plt.legend()
plt.savefig('fdg.png')


