## Code is loosely based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import os
import logging
from pathlib import Path
import datetime
import torch
import json
import numpy as np
from tqdm import tqdm
import model.pointtransformer_pose as pt_pose
from helper.ScanNetDataLoader import ScanNetDataLoader
# from helper.SimNetDataLoader import SimNetDataLoader
from helper.optimizer import RangerVA
import helper.provider as provider
from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-1269cb21-9d60-0491-7532-ba286dee143b"
torch.manual_seed(42)

def train():

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
    # NOTE: alpha, beta, gamma, delta, epsilon refer to different losses depending on 
    # the Loss used (KP, Decoder, etc).Check pointtransformer_pose.py for details.
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
            'num_keypoints': 40 # must match the number of keypoints in the dataset loaded from SimNetDataLoader
    }

    ## Create LogDir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pose_estimation')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)

    with open(str(experiment_dir) + "/config.txt", "w") as f:
        f.write(str(config))
        f.close()

    ## logger (for hyperparameter config)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler = logging.FileHandler(f"{experiment_dir}/logs.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('Hyperparameters:')
    log_string(config)
 
    # Create DataLoader
    data_path = 'data/ScanNet/ScanNet_train'
    cad_keypoint_file = 'data/cad_keypoints_scaled.txt'
    cad_pc_file = "data/YP_cloud.txt"
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

    # Save filenames used in train and test splits
    train_file = "train_files.txt"
    test_file = "test_files.txt"

    with open(train_file, "w") as f_train, open(test_file, "w") as f_test:
        for i in train_ds.indices:
            f_train.write(f"{dataset.data_paths[i][0]} {dataset.data_paths[i][1]}\n")  # Save point and pose file paths
        for i in test_ds.indices:
            f_test.write(f"{dataset.data_paths[i][0]} {dataset.data_paths[i][1]}\n")  # Save point and pose file paths

    print(f"Train set saved to {train_file}, Test set saved to {test_file}")

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

    # model = pt_pose.SortNet(128/home/karlsimon/point-transformer/log/pose_estimation/2025-02-16_18-53,6, top_k=64).cuda()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    from helper.summary import summary
    #summary(model, input_data=[(1, 128, 1024),(6, 1024)])

    # Instaed of letting summary() create the batch size, manually create dummy inputs
    dummy_input = torch.randn(2, 3, 1024).cuda()
    dummy_centroid = torch.zeros(2, 3).cuda()  # Centroid (batch, 3)
    if config['unit_sphere']:
        dummy_scale = torch.ones(2, 1).cuda()  # Scale factor (batch, 1)
    else:
        dummy_scale = torch.tensor([[2.0], [3.0]]).cuda()  # Scale factor (batch, 1)

    summary(model, input_data=[dummy_input, dummy_centroid, dummy_scale])


    # from pytictoc import TicToc

    # t = TicToc()

    # t.tic()
    # for i in range(100):
        
    #     a = torch.zeros(2, 1, 128, 1024).cuda()
    #     b = torch.zeros(2, 6, 1024).cuda()
    #     out = model(a, b)
    # t.toc()

    # UNCOMMENT TO CHECK THE MODEL
    # exit()

    # alpha for translation loss, beta for rotation loss
    # pose_criterion = pt_pose.PoseLoss(config['alpha'], config['beta']).cuda()  # Loss just used for logging
    pose_criterion = pt_pose.DecoderLoss(config['alpha'], config['beta'], config['gamma'], config['delta'], config['epsilon']).cuda()  # Loss just used for logging

    ## Create optimizer
    optimizer = None
    if config['optimizer'] == 'RangerVA':
        optimizer = RangerVA(model.parameters(), 
                            lr=config['lr'], 
                            weight_decay=config['decay_rate'])
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=config['decay_rate']
    )
    
    global_epoch = 0
    global_step = 0
    best_loss = float("inf")
            
    ## Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    for epoch in range(config['epochs']):
        log_string(f"Epoch {epoch}/{config['epochs']}")

        scheduler.step()
        total_loss = 0.0

        ## Train
        for batch_idx, data in enumerate(tqdm(train_dl, total=len(train_dl), smoothing=0.9)):
            points, gt_pose , keypoints, centroid, scale = data
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            points = points.cuda()
            points = points.transpose(1, 2) # points should have [B, C, N] format
            gt_pose = gt_pose.cuda() # gt_pose format: [qw, qx, qy, qz, tx, ty, tz]
            centroid = centroid.cuda()
            scale = scale.cuda()
            gt_rotation = gt_pose[:, :4]  # Ground-truth quaternion (B,4)
            gt_translation = gt_pose[:, 4:]  # Ground-truth translation (B,3)

            # Process keypoints (keypoints normalized)
            keypoints = keypoints.cuda() # Shape: [11, config['num_keypoints'], 5]
            gt_kp = keypoints[:, :, :3]  # Extract XYZ coordinates → [B, config['num_keypoints'], 3]
            # gt_sec = torch.argmax(keypoints[:, :, 3:], dim=-1)  # [B, config['num_keypoints']]

            optimizer.zero_grad()
            model.train()

            pred_kp, pred_R, pred_t = model(points, centroid, scale)

            use_pose_loss = 20 #epoch after which the pose_loss is used in DecoderLoss
            loss = pose_criterion(pred_kp, gt_kp, pred_R, gt_rotation, pred_t, gt_translation)
            if torch.isnan(loss):
                print(f"Epoch {epoch}, Batch {batch_idx}: NaN loss detected!")
                # print(f"pred_r: {pred_r}")
                # print(f"pred_t: {pred_t}")
                print(f"gt_rotation: {gt_rotation}")
                print(f"gt_translation: {gt_translation}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Limits gradient size
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)  # Log to TensorBoard

            global_step += 1

        avg_loss = total_loss / len(train_dl)
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        log_string(f"Train Loss: {avg_loss:.6f}")


        ## Validation
        with torch.no_grad():
            total_val_loss = 0.0
            for data in test_dl:
                points, gt_pose, keypoints, centroid, scale = data

                points = points.cuda()
                points = points.transpose(1, 2) # points should have [B, C, N] format
                gt_pose = gt_pose.cuda()
                centroid = centroid.cuda()
                scale = scale.cuda()

                model.eval()
                pred_kp, pred_R, pred_t = model(points, centroid, scale)

                gt_rotation = gt_pose[:, :4]
                gt_translation = gt_pose[:, 4:]

                # Process keypoints
                keypoints = keypoints.cuda()
                gt_kp = keypoints[:, :, :3]  # Extract XYZ coordinates → [B, config['num_keypoints'], 3]
                # gt_sec = torch.argmax(keypoints[:, :, 3:], dim=-1)  # [B, config['num_keypoints']]

                loss = pose_criterion(pred_kp, gt_kp, pred_R, gt_rotation, pred_t, gt_translation)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(test_dl)
            log_string(f"Validation Loss: {avg_val_loss:.6f}")

            # Save the best model based on lowest validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                savepath = str(experiment_dir) + '/best_model.pth'
                log_string(f"Saving best model at {savepath}")

                state = {
                    'epoch': epoch + 1,
                    'loss': avg_val_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1

    writer.close()
    

if __name__ == '__main__':
    writer = SummaryWriter(log_dir="runs/pose_training")

    train()