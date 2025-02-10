## Code is loosely based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import os
import logging
from pathlib import Path
import datetime
import torch
import numpy as np
from tqdm import tqdm
import model.pointtransformer_pose as pt_pose
from helper.ScanNetDataLoader import ScanNetDataLoader
from helper.SimNetDataLoader import SimNetDataLoader
from helper.optimizer import RangerVA
import helper.provider as provider
from torch.utils.tensorboard import SummaryWriter

def train():

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
            'lr': 0.0005,
            'decay_rate': 1e-06,
            'epochs': 100,
            'dropout': 0.4,
            'M': 4,
            'K': 64,
            'd_m': 512,
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
    # # UNCOMMENT FOR ScanNet
    # data_path = 'data/ScanNet'
    # dataset = ScanNetDataLoader(root=data_path, npoint=config['num_points'], label_channel=config['use_labels'])

    # UNCOMMENT FOR SimNet
    data_path = 'data/SimNet'
    dataset = SimNetDataLoader(root=data_path, npoint=config['num_points'], label_channel=config['use_labels'])

    # Define train-test split ratio
    train_size = int(0.95 * len(dataset))  # 95% train, 5% test
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=8)
 
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    ## Create Point Transformer model
    model = pt_pose.Point_Transformer(config).cuda()
    # model = pt_pose.SortNet(128,6, top_k=64).cuda()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    from helper.summary import summary
    #summary(model, input_data=[(1, 128, 1024),(6, 1024)])

    # Instaed of letting summary() create the batch size, manually create dummy inputs
    dummy_input = torch.randn(2, 3, 1024).cuda()
    dummy_centroid = torch.zeros(1, 3).cuda()  # Centroid (batch, 3)
    dummy_scale = torch.ones(1, 1).cuda()  # Scale factor (batch, 1)

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

    pose_criterion = pt_pose.PoseLoss(alpha=10).cuda()
    
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
            points, gt_pose , centroid, scale = data
            points = points.cuda()
            points = points.transpose(1, 2) # points should have [B, C, N] format
            gt_pose = gt_pose.cuda() # gt_pose format: [qw, qx, qy, qz, tx, ty, tz]
            centroid = centroid.cuda()
            scale = scale.cuda()
            gt_rotation = gt_pose[:, :4]  # Ground-truth quaternion (B,4)
            gt_translation = gt_pose[:, 4:]  # Ground-truth translation (B,3)

            optimizer.zero_grad()
            model.train()

            pred_r, pred_t = model(points, centroid, scale)
            loss = pose_criterion(pred_r, gt_rotation, pred_t, gt_translation)
            if torch.isnan(loss):
                print(f"Epoch {epoch}, Batch {batch_idx}: NaN loss detected!")
                print(f"pred_r: {pred_r}")
                print(f"pred_t: {pred_t}")
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
                points, gt_pose, centroid, scale = data

                points = points.cuda()
                points = points.transpose(1, 2) # points should have [B, C, N] format
                gt_pose = gt_pose.cuda()
                centroid = centroid.cuda()
                scale = scale.cuda()

                model.eval()
                pred_r, pred_t = model(points, centroid, scale)

                gt_rotation = gt_pose[:, :4]
                gt_translation = gt_pose[:, 4:]

                loss = pose_criterion(pred_r, gt_rotation, pred_t, gt_translation)
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


def test(model, loader, config):
    mean_correct = []
    class_acc = np.zeros((config['num_classes'],3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-1269cb21-9d60-0491-7532-ba286dee143b"
    writer = SummaryWriter(log_dir="runs/pose_training")

    train()
