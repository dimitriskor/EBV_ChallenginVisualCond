import random
import os
import subprocess

# Define the directory where the output files will be saved
output_directory = "../results"

# Define the base command
base_command = (
    "python3 inference_small_short_inf.py "
    "--checkpoint_path ../reconstruction_model.pth "
    "--device 1 "
    "--events_file_path /mnt/raid0a/Dimitris/DSEC/event_recordings/{output_folder}/events/left/events.h5 "
    "--output_folder ../final_res/{output_folder} "
    "--voxel_method t_seconds "
    "--t 25000 "
    "--sliding_window_t 0 "
    "--folder_name {output_folder}"
)
# Loop through files in the output directory
rec_names = list(os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/'))
# random.shuffle(rec_names)
rec_names.sort()
for i in range(1):
    for out_file in rec_names:
        # if os.path.isdir('../results_short_small_inf/'+out_file):
        #     continue
        # if 'interlaken_01' in out_file or 'interlaken_00_b' in out_file or 'zurich_city_12' in out_file or 'zurich_city_09' in out_file or 'zurich_city_10' in out_file or 'zurich_city_04_f' in out_file or 'zurich_city_02' in out_file or 'zurich_city_03' in out_file:
        #     continue
        if 'zurich_city_04_f' not in out_file:
            continue
        output_path = f"{out_file}"  # Construct the output folder path
        command = base_command.format(output_folder=out_file)  # Format the command
        print(f"Running command: {command}")  # Debugging: print the command

        # Run the command
        subprocess.run(command, shell=True, check=True)  # Use `check=True` to raise an error if the command fails
