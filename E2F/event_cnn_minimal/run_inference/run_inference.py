import os
import subprocess

# Define the directory where the output files will be saved
output_directory = "../results"

# Define the base command
base_command = (
    "python3 inference_huge.py "
    "--checkpoint_path ../reconstruction_model.pth "
    "--device 0 "
    "--events_file_path /mnt/raid0a/Dimitris/DSEC/event_recordings/{output_folder}/events/left/events.h5 "
    "--output_folder ../results/{output_folder} "
    "--voxel_method t_seconds "
    "--t 200000 "
    "--sliding_window_t 150000 "
    "--folder_name {output_folder}"
)
# Loop through files in the output directory
for i in range(20):
    for out_file in os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/'):
        if 'interlaken_01' in out_file or 'interlaken_00_b' in out_file or 'zurich_city_12' in out_file or 'zurich_city_09' in out_file or 'zurich_city_10' in out_file or 'zurich_city_04_f' in out_file or 'zurich_city_02' in out_file :
            continue
        output_path = f"{out_file}"  # Construct the output folder path
        command = base_command.format(output_folder=out_file)  # Format the command
        print(f"Running command: {command}")  # Debugging: print the command

        # Run the command
        subprocess.run(command, shell=True, check=True)  # Use `check=True` to raise an error if the command fails
