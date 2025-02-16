## Code is loosely based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import os
import logging
from pathlib import Path
import datetime
import torch
import numpy as np
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation as R
import model.pointtransformer_pose as pt_pose
from helper.ScanNetDataLoader import ScanNetDataLoader
from helper.SimNetDataLoader import SimNetDataLoader
from helper.optimizer import RangerVA
import helper.provider as provider

torch.manual_seed(42)

def test():

    # To check CUDA and PyTorch installation: $ conda list | grep 'pytorch\|cudatoolkit'
    device_id = 0  # Change this to 1 to use the second GPU
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
            'lr': 0.0005,
            'decay_rate': 1e-06,
            'epochs': 100,
            'dropout': 0.4,
            'M': 4,
            'K': 64,
            'd_m': 512,
            'alpha': 10,
            'beta': 1,
            'radius_max_points': 32,
            'radius': 0.2
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
 
    data_path = 'data/SimNet'
    dataset = SimNetDataLoader(root=data_path, npoint=config['num_points'], label_channel=config['use_labels'])

    # Define train-test split ratio
    train_size = int(0.95 * len(dataset))  # 95% train, 5% test
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"First 5 train samples: {[dataset.data_paths[i] for i in train_ds.indices[:5]]}")
    print(f"First 5 test samples: {[dataset.data_paths[i] for i in test_ds.indices[:5]]}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=8)
 
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    ## Create Point Transformer model
    model = pt_pose.Point_Transformer(config).cuda()

    from helper.summary import summary
    dummy_input = torch.randn(2, 3, 1024).cuda()
    dummy_centroid = torch.zeros(1, 3).cuda()  # Centroid (batch, 3)
    dummy_scale = torch.ones(1, 1).cuda()  # Scale factor (batch, 1)

    summary(model, input_data=[dummy_input, dummy_centroid, dummy_scale])

    # Load saved model
    checkpoint_path = "/home/karlsimon/point-transformer/log/pose_estimation/2025-02-13_16-30/best_model.pth"
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"]) #load the weights
    model.eval()
    print(f"Loaded best model from {checkpoint_path}, trained until epoch {checkpoint['epoch']}")

    pose_criterion = pt_pose.PoseLoss(config['alpha'], config['beta']).cuda() #Loss just used for logging

    result_data = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dl):
            points, gt_pose, centroid, scale = data

            points = points.cuda()
            points = points.transpose(1, 2)  # points should have [B, C, N] format
            gt_pose = gt_pose.cuda()
            centroid = centroid.cuda()
            scale = scale.cuda()

            model.eval()
            pred_r, pred_t = model(points, centroid, scale)

            gt_rotation = gt_pose[:, :4] #still in WXYZ (as dataset stores it)
            gt_translation = gt_pose[:, 4:]

            loss = pose_criterion(pred_r, gt_rotation, pred_t, gt_translation).item()  # Computed for the batch

            # Convert angle-axis to quaternion for logging
            pred_r_np = pred_r.cpu().numpy()
            pred_quat = np.array([R.from_rotvec(r).as_quat() for r in pred_r_np])  # NOTE: scipy pred_quat has XYZW format!
            pred_translation = pred_t.cpu().numpy()
            gt_quat_xyzw = np.roll(gt_rotation.cpu().numpy(), shift=-1, axis=1)  # Convert WXYZ → XYZW for writing to results.json
            print("gt_quat_xyzw", gt_quat_xyzw)
            gt_quat = gt_quat_xyzw #just for now
            gt_translation = gt_translation.cpu().numpy()

            # Retrieve file names for this batch
            batch_start_idx = batch_idx * config['batch_size']
            batch_file_names = [dataset.data_paths[test_ds.indices[i]][0] for i in range(batch_start_idx, batch_start_idx + len(gt_rotation))]

            # Store inference results in a structured format
            for i in range(len(gt_rotation)):
                result_data.append({
                    "file": batch_file_names[i],
                    "gt_rotation": gt_quat[i].tolist(),
                    "gt_translation": gt_translation[i].tolist(),
                    "pred_rotation": pred_quat[i].tolist(),
                    "pred_translation": pred_translation[i].tolist(),
                    "loss": loss
                })

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
