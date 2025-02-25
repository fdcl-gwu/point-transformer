import os
import shutil

# Define paths
dir1 = "/home/karlsimon/point-transformer/data/SimNet/gazebo_pc_record_full_12_42_1024"
dir2 = "/home/karlsimon/point-transformer/data/SimNet15/gazebo_pc_record_full_15_41_1024"
merged_dir = "/home/karlsimon/point-transformer/data/SimNet15/gazebo_pc_record_full_12_15_merged_1024"

# Create merged directories if they don't exist
os.makedirs(os.path.join(merged_dir, "clouds"), exist_ok=True)
os.makedirs(os.path.join(merged_dir, "poses"), exist_ok=True)

# Get the list of files in dir1 and determine the last number
dir1_files = sorted(os.listdir(os.path.join(dir1, "clouds")))
last_num_dir1 = int(dir1_files[-1].split(".")[0])  # Last number in dir1

# Move dir1 files to merged directory
for file in dir1_files:
    shutil.copy(os.path.join(dir1, "clouds", file), os.path.join(merged_dir, "clouds", file))
    shutil.copy(os.path.join(dir1, "poses", file), os.path.join(merged_dir, "poses", file))

# Get the list of files in dir2 and rename with offset
dir2_files = sorted(os.listdir(os.path.join(dir2, "clouds")))

for file in dir2_files:
    num = int(file.split(".")[0])
    new_num = num + last_num_dir1 + 1  # Offset numbering
    new_filename = f"{new_num:06d}.txt"
    
    shutil.copy(os.path.join(dir2, "clouds", file), os.path.join(merged_dir, "clouds", new_filename))
    shutil.copy(os.path.join(dir2, "poses", file), os.path.join(merged_dir, "poses", new_filename))

print("Merging complete.")
