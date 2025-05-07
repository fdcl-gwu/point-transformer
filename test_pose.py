## Code is loosely based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import os
import logging
from pathlib import Path
import datetime
import time
import torch
import numpy as np
from tqdm import tqdm
import json
import open3d as o3d
import small_gicp
from scipy.spatial.transform import Rotation as R
import model.pointtransformer_pose as pt_pose
from helper.ScanNetDataLoader import ScanNetDataLoader
# from helper.SimNetDataLoader import SimNetDataLoader
from helper.optimizer import RangerVA
import helper.provider as provider

torch.manual_seed(42)

def test():

    # To check CUDA and PyTorch installation: $ conda list | grep 'pytorch\|cudatoolkit'
    device_id = 1  # Change this to 1 to use the second GPU
    torch.cuda.set_device(device_id)

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available. Using CPU.")

    def log_string(str):
        logger.info(str)
        print(str)

    ## Hyperparameters
    config = {'num_points' : 1024,
            'batch_size': 11,
            'use_labels': False,
            'optimizer': 'RangerVA',
            'lr': 0.001,
            'decay_rate': 1e-06,
            'epochs': 100,
            'dropout': 0.4,
            'M': 4,
            'K': 64,
            'd_m': 512,
            'alpha': 5,
            'beta': 0,
            'gamma': 7,
            'delta': 0,
            'epsilon': 1,
            'radius_max_points': 32,
            'radius': 0.2,
            'unit_sphere': True,
            'num_keypoints': 40 # must match the number of keypoints in the dataset loaded from ScanNetDataLoader
    }

    # Create inference log directory
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    inference_dir = Path('./log/pose_est_inference/')
    inference_dir.mkdir(exist_ok=True)
    inference_dir = inference_dir.joinpath(timestr)
    inference_dir.mkdir(exist_ok=True)

    # Create logger for inference
    logger = logging.getLogger("Inference")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler = logging.FileHandler(f"{inference_dir}/inference_logs.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_inference(pred_quat, pred_translation, gt_quat, gt_translation, loss):
        logger.info(f"Predicted Rotation (Quaternion): {pred_quat}")
        logger.info(f"Predicted Translation: {pred_translation}")
        logger.info(f"Ground Truth Rotation (Quaternion): {gt_quat}")
        logger.info(f"Ground Truth Translation: {gt_translation}")
        logger.info(f"Pose Estimation Loss: {loss:.6f}\n")
 
    data_path = 'data/ScanNet'
    cad_keypoint_file = 'data/ship_keypoints_40_cfg_st_dg_few.txt'
    cad_pc_file = "data/yp_complete_cloud_less_dense.txt"
    dataset = ScanNetDataLoader(root=data_path, npoint=config['num_points'], label_channel=config['use_labels'], unit_sphere=config['unit_sphere'])

    # Define train-test split ratio (NOT RANDOM)
    total_samples = len(dataset)
    train_cutoff = int(0.95 * total_samples)

    train_indices = list(range(train_cutoff))
    test_indices = list(range(train_cutoff, total_samples))

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    print(f"First 5 train samples: {[dataset.data_paths[i] for i in train_ds.indices[:5]]}")
    print(f"First 5 test samples: {[dataset.data_paths[i] for i in test_ds.indices[:5]]}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0, drop_last=True)
 
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # CAD keypoints — use raw coordinates for decoder queries
    cad_kp = torch.tensor(np.loadtxt(cad_keypoint_file, dtype=np.float32))  # [40, 3]
    cad_pc = torch.tensor(np.loadtxt(cad_pc_file, dtype=np.float32))

    # Compute scale/centroid from cad_kp (not cad_pc)
    cad_kp_centroid = cad_kp.mean(dim=0)
    cad_kp_scale = cad_kp.norm(dim=1).max()

    # Do NOT normalize cad_kp here
    cad_kp = cad_kp.cuda()
    cad_pc = cad_pc.cuda()

    model = pt_pose.Point_Transformer(
        config,
        cad_kp=cad_kp,                     # unnormalized
        cad_centroid=cad_kp_centroid.cuda(),
        cad_scale=cad_kp_scale.cuda()
    ).cuda()

    from helper.summary import summary
    #summary(model, input_data=[(1, 128, 1024),(6, 1024)])
    dummy_input = torch.randn(2, 3, 1024).cuda()
    dummy_centroid = torch.zeros(2, 3).cuda()  # Centroid (batch, 3)
    if config['unit_sphere']:
        dummy_scale = torch.ones(2, 1).cuda()  # Scale factor (batch, 1)
    else:
        dummy_scale = torch.tensor([[2.0], [3.0]]).cuda()  # Scale factor (batch, 1)

    summary(model, input_data=[dummy_input, dummy_centroid, dummy_scale])

    # Load saved model
    checkpoint_path = "/home/karlsimon/point-transformer/log/pose_estimation/2025-05-05_16-44/best_model.pth"
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"]) #load the weights
    model.eval()
    print(f"Loaded best model from {checkpoint_path}, trained until epoch {checkpoint['epoch']}")

    pose_criterion = pt_pose.DecoderLoss(config['alpha'], config['beta'], config['gamma'], config['delta'], config['epsilon']).cuda()  # Loss just used for logging

    result_data = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dl):
            points, gt_pose, keypoints, centroid, scale = data

            points = points.cuda()
            points = points.transpose(1, 2)  # points should have [B, C, N] format
            gt_pose = gt_pose.cuda()
            centroid = centroid.cuda()
            scale = scale.cuda()

            model.eval()
            pred_kp, pred_R, pred_t = model(points, centroid, scale)

            gt_rotation = gt_pose[:, :4] #still in WXYZ (as dataset stores it)
            gt_translation = gt_pose[:, 4:]

            # Convert rotations to rotation numpy matrices
            gt_quat_wxyz = gt_rotation.cpu().numpy()  # shape [B, 4]
            gt_rot_matrices = R.from_quat(gt_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # convert WXYZ → XYZW before conversion

            pred_rot_matrices = pred_R.cpu().numpy()  # assuming already rotation matrices
            pred_translation_np = pred_t.cpu().numpy()
            gt_translation_np = gt_translation.cpu().numpy()

            # Process keypoints
            keypoints = keypoints.cuda()
            gt_kp = keypoints[:, :, :3]  # Extract XYZ coordinates → [B, config['num_keypoints'], 3
            # gt_sec = torch.argmax(keypoints[:, :, 3:], dim=-1)  # [B, config['num_keypoints']]

            loss = pose_criterion(pred_kp, gt_kp, pred_R, gt_rotation, pred_t, gt_translation)
            # # Convert angle-axis to quaternion for logging
            # pred_r_np = pred_r.cpu().numpy()
            # pred_quat = np.array([R.from_rotvec(r).as_quat() for r in pred_r_np])  # NOTE: scipy pred_quat has XYZW format!
            # pred_translation = pred_t.cpu().numpy()
            # gt_quat_xyzw = np.roll(gt_rotation.cpu().numpy(), shift=-1, axis=1)  # Convert WXYZ → XYZW for writing to results.json
            # print("gt_quat_xyzw", gt_quat_xyzw)
            # gt_quat = gt_quat_xyzw #just for now
            # gt_translation = gt_translation.cpu().numpy()

            pred_kp_np = pred_kp.cpu().numpy()
            # pred_sec_np = torch.argmax(pred_sec, dim=-1).cpu().numpy()
            gt_kp_np = gt_kp.cpu().numpy()
            # gt_sec_np = gt_sec.cpu().numpy()
            # write to results.json

            # Retrieve file names for this batch
            batch_start_idx = batch_idx * config['batch_size']
            batch_file_names = [dataset.data_paths[test_ds.indices[i]][0] for i in range(batch_start_idx, batch_start_idx + len(gt_rotation))]
            
            for i in range(len(gt_rotation)):
                # Create base entry first
                file_name = batch_file_names[i]
                entry = {
                    "file": file_name,
                    "gt_kp": gt_kp_np[i].tolist(),
                    "pred_kp": pred_kp_np[i].tolist(),
                    "gt_rotation": gt_rot_matrices[i].tolist(),
                    "gt_translation": gt_translation_np[i].tolist(),
                    "pred_rotation": pred_rot_matrices[i].tolist(),
                    "pred_translation": pred_translation_np[i].tolist(),
                    "scale": scale[i].cpu().numpy().tolist(),
                    "centroid": centroid[i].cpu().numpy().tolist(),
                    "points": points[i].transpose(0, 1).cpu().numpy().tolist()
                }

                # Run small_gicp refinement
                start_time = time.time()
                scan_points = np.array(points[i].transpose(0, 1).cpu().numpy())
                centroid_np = centroid[i].cpu().numpy()
                scale_np = scale[i].cpu().numpy()

                # Unnormalize scan points
                scan_points_unnorm = scan_points * scale_np + centroid_np
                ship_points = np.loadtxt(cad_pc_file)

                T_init = np.eye(4)
                T_init[:3, :3] = pred_rot_matrices[i]
                T_init[:3, 3] = pred_translation_np[i]
                print(f"Initial T_target_source:\n{T_init}")

                result = small_gicp.align(
                    target_points=ship_points,
                    source_points=scan_points_unnorm,
                    init_T_target_source=T_init,
                    registration_type='GICP',
                    downsampling_resolution=0.05,
                    max_correspondence_distance=1.0,
                    num_threads=4,
                    max_iterations=5,
                )

                elapsed = time.time() - start_time
                print(f"Sample {i + 1}: took {elapsed:.3f} seconds")

                # Add refinement results
                T_refined = result.T_target_source
                print("Refined T_target_source:\n", T_refined)
                entry["refined_rotation"] = T_refined[:3, :3].tolist()
                entry["refined_translation"] = T_refined[:3, 3].tolist()
                entry["gicp_converged"] = result.converged
                entry["gicp_iterations"] = result.iterations

                result_data.append(entry)

            print(f"Processed batch {batch_idx + 1}/{len(test_dl)}")

    # Save to JSON file
    result_filename = inference_dir.joinpath("results.json")
    with open(result_filename, "w") as f:
        json.dump(result_data, f, indent=4)

    print(f"Results saved to {result_filename}")    

    # TODO 12.02.25: determine better evaluation metric
    # FOR NOW: send to MAC for visualization
    

if __name__ == '__main__':
    
    test()

