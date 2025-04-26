import os
import subprocess

# Define the directory where the output files will be saved
output_directory = "../results"

# Define the base command
base_command = (
    "python3 inference.py "
    "--checkpoint_path ../etnet.pth "
    "--device 1 "
    "--events_file_path /mnt/raid0a/Dimitris/DSEC/event_recordings/{output_folder}/events/left/events.h5 "
    "--output_folder ../results_etnet/{output_folder} "
    "--voxel_method t_seconds "
    "--t 50000 "
    "--sliding_window_t 1 "
)
# Loop through files in the output directory
for out_file in os.listdir('/mnt/raid0a/Dimitris/DSEC/event_recordings/'):
    output_path = f"{out_file}"  # Construct the output folder path
    command = base_command.format(output_folder=output_path)  # Format the command
    print(f"Running command: {command}")  # Debugging: print the command

    # Run the command
    subprocess.run(command, shell=True, check=True)  # Use `check=True` to raise an error if the command fails
