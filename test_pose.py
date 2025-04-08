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
            'epochs': 70,
            'dropout': 0.4,
            'M': 4,
            'K': 64,
            'd_m': 512,
            'alpha': 2,
            'beta': 5,
            'gamma': 3,
            'delta': 0.001,
            'epsilon': 1,
            'radius_max_points': 32,
            'radius': 0.2,
            'unit_sphere': True,
            'num_keypoints': 40 # must match the number of keypoints in the dataset loaded from SimNetDataLoader
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
 
    data_path = 'data/SimNet_close'
    cad_keypoint_file = 'data/cad_keypoints_40_cfg_st_dg_few.txt'
    cad_pc_file = "data/rotated_Ship_copy_downsampled_neg05.txt"
    dataset = SimNetDataLoader(root=data_path, npoint=config['num_points'], label_channel=config['use_labels'], unit_sphere=config['unit_sphere'])

    # Define train-test split ratio
    train_size = int(0.95 * len(dataset))  # 95% train, 5% test
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"First 5 train samples: {[dataset.data_paths[i] for i in train_ds.indices[:5]]}")
    print(f"First 5 test samples: {[dataset.data_paths[i] for i in test_ds.indices[:5]]}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=8)
 
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Load CAD points, keypoints and normalize ONCE
    cad_kp = torch.tensor(np.loadtxt(cad_keypoint_file, dtype=np.float32))  # on CPU for now
    cad_pc = torch.tensor(np.loadtxt(cad_pc_file, dtype=np.float32))

    cad_centroid = cad_pc.mean(dim=0) # for centering whole ship cloud (not ship keypoints cloud)
    cad_pc = cad_pc - cad_centroid
    cad_scale = cad_pc.norm(dim=1).max()

    if config['unit_sphere']:
        cad_pc = cad_pc / cad_scale
        cad_kp = (cad_kp - cad_centroid) / cad_scale
        print("cad_centroid: ", cad_centroid, " and cad_scale: ", cad_scale)

    cad_kp = cad_kp.cuda()
    cad_pc = cad_pc.cuda()

    ## Create Point Transformer model
    model = pt_pose.Point_Transformer(
        config,
        cad_kp=cad_kp.cuda(),
        cad_centroid=cad_centroid.cuda(),
        cad_scale=cad_scale.cuda()
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
    checkpoint_path = "/home/karlsimon/point-transformer/log/pose_estimation/2025-04-08_12-00/best_model.pth"
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
            for i in range(len(gt_rotation)):
                result_data.append({
                    "gt_kp": gt_kp_np[i].tolist(),
                    "pred_kp": pred_kp_np[i].tolist(),
                    "gt_rotation": gt_rot_matrices[i].tolist(),
                    "gt_translation": gt_translation_np[i].tolist(),
                    "pred_rotation": pred_rot_matrices[i].tolist(),
                    "pred_translation": pred_translation_np[i].tolist(),
                    # "loss": loss.item()
                })


            # Retrieve file names for this batch
            batch_start_idx = batch_idx * config['batch_size']
            batch_file_names = [dataset.data_paths[test_ds.indices[i]][0] for i in range(batch_start_idx, batch_start_idx + len(gt_rotation))]

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

