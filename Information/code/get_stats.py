import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

model_types = ['11n', '11x', 'v3']

total_stats = {'frames': [],
            'person' : [],
            'car' : [],
            'bicycle' : [],
            'motorcycle' : [],
            'bus' : [],
            'truck' : [],
            'traffic_light' : [],
            'stop_sign' : [],
            'bicycle' : [],
            }
        

for model_type in model_types:

    model_stats = {'frames': 0,
                'person' : 0,
                'car' : 0,
                'bicycle' : 0,
                'motorcycle' : 0,
                'bus' : 0,
                'truck' : 0,
                'traffic_light' : 0,
                'stop_sign' : 0,
                'bicycle' : 0,
                }
    stats = list(os.listdir(f'../statistics/glare/{model_type}/'))
    prefix = f'../statistics/glare/{model_type}/'


    for file in stats:
        with open(prefix+file, 'rb') as f:
            data = pickle.load(f)

            for k in model_stats.keys():
                model_stats[k] += data[k]

    for k in model_stats.keys():
        total_stats[k].append(model_stats[k])

column_labels = ['Model v11 tiny', 'Model v11 huge', 'Model v3']
row_labels = list(total_stats.keys())
table_values = [v for v in total_stats.values()]  # Create a list of lists for the values

# Create a figure and axis
fig, ax = plt.subplots()

# Hide the axes (not needed for table)
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=table_values,
                rowLabels=row_labels,
                 colLabels=column_labels,
                cellLoc='center',
                loc='center')
table.scale(.8, 1.5)
# plt.title(f'Model {model_type}')
plt.savefig('table.png')