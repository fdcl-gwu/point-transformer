## Implementation borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import json
warnings.filterwarnings('ignore')


def load_pose_file(filepath, to_quaternion=True):
    """Load pose data from a JSON-like format and convert to quaternion (if needed)."""
    with open(filepath, 'r') as f:
        pose_data = json.load(f) # The output from stl_cloud_processing is in json format
    
    # Extract translation and quaternion
    translation = np.array([pose_data["x"], pose_data["y"], pose_data["z"]], dtype=np.float32)
    quaternion = np.array([pose_data["qx"], pose_data["qy"], pose_data["qz"], pose_data["qw"]], dtype=np.float32)
    
    return np.hstack((quaternion, translation))  # [qx, qy, qz, qw, tx, ty, tz]
    
# TODO: choose better way of normalizing
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class SimNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, uniform=False, label_channel=False, cache_size=15000):
        self.root = root # /data/SimNet
        self.npoints = npoint # 1024
        self.uniform = uniform
        self.label_channel = label_channel
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (points, poses) tuple

        self.data_paths = []
        scan_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))] # scan_dirs will look like this: ['/data/SimNet/datset1', '/data/SimNet/datset2', ...]

        for scan_dir in scan_dirs:
            points_dir = os.path.join(scan_dir, "clouds")
            poses_dir = os.path.join(scan_dir, "poses")

            point_files = sorted([f for f in os.listdir(points_dir)])
            pose_files = sorted([f for f in os.listdir(poses_dir)])

        # Checks if pose files exist for all point files
        for point_file in point_files:
            pose_file = point_file #since the pose file has the same name as the point file
            if pose_file in pose_files:
                self.data_paths.append((os.path.join(points_dir, point_file), os.path.join(poses_dir, pose_file)))
            else:
                print(f"Warning: No pose file found for {point_file}")

        print(f"Loaded {len(self.data_paths)} samples from {root}.")


    def __len__(self):
        return len(self.data_paths)
    

    def _get_item(self, index):
        cloud_path, pose_path = self.data_paths[index]

        cache_key = f"{cloud_path}_{pose_path}" # In the case that filenames are shared between directories

        if cache_key in self.cache:
            points, poses = self.cache[cache_key]
        else:
            # Load the point cloud from the .txt file
            point_cloud = np.loadtxt(cloud_path).astype(np.float32) 

            # NOTE: no FPS for Gazebo scans. done in dataset preprocessing
            point_cloud = point_cloud[:self.npoints, :]

            # Optional normalization (currently commented out)
            # point_cloud[:, :3] = pc_normalize(point_cloud[:, :3])

            # Load the pose data
            pose = load_pose_file(pose_path)

            # If label_channel=False, only return xyz coordinates. Otherwise, uses xyzl with l between 0-9
            if not self.label_channel:
                point_cloud = point_cloud[:, 0:3]
                print("Not using label channel")

            # Store in cache if limit is not exceeded
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = (point_cloud, pose)

        return point_cloud, pose

        

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = SimNetDataLoader('/data/ScanNet/', uniform=False, label_channel=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    
    for points, poses in DataLoader:
        print(points.shape)  # Expected: [batch_size, 1024, 3]
        print(poses.shape)   # Expected: [batch_size, 7]