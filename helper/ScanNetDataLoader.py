## Implementation borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import open3d as o3d
warnings.filterwarnings('ignore')



def load_ply_file(filepath):
    """Load PLY file and return point cloud as a numpy array."""
    ply = o3d.io.read_point_cloud(filepath)
    return np.asarray(ply.points, dtype=np.float32)


def load_pose_file(filepath, to_quaternion=True):
    """Load 4x4 transformation matrix and convert to quaternion (if needed)."""
    pose = np.loadtxt(filepath).astype(np.float32)
    rotation_matrix = pose[:3, :3]
    translation = pose[:3, 3]

    if to_quaternion:
        quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        return np.hstack((quat, translation))  # [qx, qy, qz, qw, tx, ty, tz]
    else:
        return pose  # Return full 4x4 matrix
    
# TODO: choose better way of normalizing
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ScanNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, uniform=False, normal_channel=False, to_quaternion=True, cache_size=15000):
        self.root = root # /data/ScanNet
        self.npoints = npoint # 1024
        self.uniform = uniform
        self.normal_channel = normal_channel
        self.to_quaternion = to_quaternion
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (points, poses) tuple

        self.data_paths = []
        scan_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))] # scan_dirs will look like this: ['/data/ScanNet/08_11_yp_1', '/data/ScanNet/08_11_yp_2', ...]

        for scan_dir in scan_dirs:
            points_dir = os.path.join(scan_dir, "clouds")
            poses_dir = os.path.join(scan_dir, "poses")

            point_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.ply')])
            pose_files = sorted([f for f in os.listdir(poses_dir) if f.endswith('.txt')])

        # Checks if pose files exist for all point files
        for point_file in point_files:
            pose_file = point_file.replace('.ply', '.txt')
            if pose_file in pose_files:
                self.data_paths.append((os.path.join(points_dir, point_file), os.path.join(poses_dir, pose_file)))
            else:
                print(f"Warning: No pose file found for {point_file}")

        print(f"Loaded {len(self.data_paths)} samples from {root}.")


    def __len__(self):
        return len(self.data_paths)
    

    def _get_item(self, index):
        ply_path, pose_path = self.data_paths[index]

        cache_key = f"{ply_path}_{pose_path}" # In the case that filenames are shared between directories

        if cache_key in self.cache:
            points, poses = self.cache[cache_key]
        else:
            # Load the point cloud from the PLY file
            point_cloud = load_ply_file(ply_path)

            # Use farthest point sampling if needed
            if self.uniform:
                point_cloud = farthest_point_sample(point_cloud, self.npoints)
            else:
                point_cloud = point_cloud[:self.npoints, :]

            # Optional normalization (currently commented out)
            # point_cloud[:, :3] = pc_normalize(point_cloud[:, :3])

            # Load the pose data
            pose = load_pose_file(pose_path, self.to_quaternion)

            # If normal_channel=False, only return xyz coordinates
            if not self.normal_channel:
                point_cloud = point_cloud[:, 0:3]

            # Store in cache if limit is not exceeded
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = (point_cloud, pose)

        return point_cloud, pose

        

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ScanNetDataLoader('/data/ScanNet/', uniform=False, normal_channel=False, to_quaternion=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    
    for points, poses in DataLoader:
        print(points.shape)  # Expected: [batch_size, 1024, 3]
        print(poses.shape)   # Expected: [batch_size, 7] if quaternion, or [batch_size, 4, 4] otherwise