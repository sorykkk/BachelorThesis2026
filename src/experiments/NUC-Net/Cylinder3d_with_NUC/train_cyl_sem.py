# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

import wandb

# Auto-load WANDB_API_KEY from .env.local
_env_file = Path(__file__).resolve().parents[3] / ".env.local"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip().strip('"'))
    print(f"[wandb] Loaded env from {_env_file}")
else:
    print(f"[wandb] WARNING: {_env_file} not found")

if "WANDB_API_KEY" in os.environ:
    print("[wandb] API key found, logging will be online.")
    os.environ["WANDB_MODE"] = "online"
else:
    print("[wandb] WARNING: WANDB_API_KEY not set. Running in offline mode.")
    os.environ["WANDB_MODE"] = "offline"

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint, save_full_checkpoint, load_full_checkpoint

import warnings

warnings.filterwarnings("ignore")


def verify_nuc_api(grid_size, first_term=0.05, tolerance=0.0062):
    """Verify NUC (API) partitioning is working correctly at startup."""
    print("\n" + "="*60)
    print("[NUC/API Verification]")
    print(f"  first_term (a0) = {first_term}")
    print(f"  tolerance  (d)  = {tolerance}")
    print(f"  grid_size       = {grid_size}")

    # Simulate the API formula for a few radial distances
    # r_max = 50 - 1 = 49 (from max_volume_space[0] - min_volume_space[0])
    test_r_values = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 49.0]
    a0 = first_term
    d = tolerance
    print(f"\n  Radial distance -> NUC grid index (should be non-uniform, increasing):")
    prev_idx = -1
    all_increasing = True
    for r in test_r_values:
        discriminant = (2 * a0 + d)**2 - 8 * (d - r) * d
        if discriminant < 0:
            print(f"    r={r:5.1f}m -> WARNING: negative discriminant!")
            all_increasing = False
            continue
        idx = np.ceil(((-2 * a0 - d) + np.sqrt(discriminant)) / (2 * d))
        idx = int(np.clip(idx, 0, grid_size[0] - 1))
        print(f"    r={r:5.1f}m -> grid_idx={idx}")
        if idx <= prev_idx and r > test_r_values[0]:
            all_increasing = False
        prev_idx = idx

    # Verify first few intervals grow
    intervals = []
    for i in range(5):
        a_i = a0 + i * d
        intervals.append(a_i)
    print(f"\n  First 5 API intervals: {[f'{x:.4f}' for x in intervals]}")
    print(f"  Intervals are increasing: {all(intervals[i] < intervals[i+1] for i in range(len(intervals)-1))}")
    print(f"  Grid indices are non-decreasing: {all_increasing}")

    if all_increasing:
        print("  [OK] NUC/API partitioning looks correct!")
    else:
        print("  [WARNING] NUC/API partitioning may have issues!")
    print("="*60 + "\n")
    return all_increasing


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    num_scales = model_config['num_scales']

    model_load_path = train_hypers.get('model_load_path', '')
    model_save_path = train_hypers['model_save_path']
    checkpoint_path = train_hypers.get('checkpoint_path', model_save_path.replace('.pt', '_checkpoint.pt'))

    # --- NUC (API) Verification ---
    verify_nuc_api(grid_size, first_term=0.05, tolerance=0.0062)

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    # Resume from full checkpoint if available, otherwise load legacy weights
    resume_epoch = 0
    resume_global_iter = 0
    resume_best_val_miou = 0.0
    if os.path.exists(checkpoint_path):
        my_model, optimizer, resume_epoch, resume_global_iter, resume_best_val_miou = \
            load_full_checkpoint(checkpoint_path, my_model, optimizer, device=str(pytorch_device))
    elif os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    # --- Instance Augmentation (IA) ---
    instance_augmentor = None
    ia_db_path = train_hypers['instance_db_path']
    if os.path.exists(ia_db_path):
        from dataloader.instance_augmentation import InstanceAugmentor
        instance_augmentor = InstanceAugmentor(ia_db_path, max_paste_per_class=3)
        print(f"[IA] Instance Augmentation ENABLED from {ia_db_path}")
    else:
        print(f"[IA] WARNING: Instance database not found at {ia_db_path}")
        print(f"[IA] Run: python build_instance_db.py --data_path {train_dataloader_config['data_path']}")
        print(f"[IA] Instance Augmentation DISABLED (will lose ~2% mIoU)")

    # --- NUMA ---
    print(f"[NUMA] Non-Uniform Multi-scale Aggregation: num_scales={num_scales}")

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  instance_augmentor=instance_augmentor,
                                                                  num_scales=num_scales)
    # Initialize wandb
    wandb.init(
        entity='soryn-besleaga-universitatea-politehnica-timi',
        project='NUC-Net-SemanticKITTI',
        config={
            'architecture': 'Cylinder3D_NUC',
            'dataset': 'SemanticKITTI',
            'config_file': config_path,
            'learning_rate': train_hypers['learning_rate'],
            'max_epochs': train_hypers['max_num_epochs'],
            'eval_every_n_steps': train_hypers['eval_every_n_steps'],
            'train_batch_size': train_batch_size,
            'val_batch_size': val_batch_size,
            'grid_size': grid_size,
            'num_class': num_class,
            'num_scales': num_scales,
            'instance_augmentation': instance_augmentor is not None,
            'first_term': 0.05,
            'tolerance': 0.0062,
        },
        name='train-semantickitti-singlescan-NUC-NUMA-IA',
        resume='allow',
    )

    # training
    epoch = resume_epoch
    best_val_miou = resume_best_val_miou
    my_model.train()
    global_iter = resume_global_iter
    check_iter = train_hypers['eval_every_n_steps']

    # Gradient accumulation: physical batch_size=2 * accumulation_steps=4 = effective batch_size=8
    accumulation_steps = 4

    print(f"\n[Training Config]")
    print(f"  Physical batch size: {train_batch_size}")
    print(f"  Accumulation steps:  {accumulation_steps}")
    print(f"  Effective batch size: {train_batch_size * accumulation_steps}")
    print(f"  Learning rate: {train_hypers['learning_rate']}")
    print(f"  Max epochs: {train_hypers['max_num_epochs']}")
    print(f"  Components: NUC(API) + NUMA(scales={num_scales}) + IA({'ON' if instance_augmentor else 'OFF'})")
    print()

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        optimizer.zero_grad()
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea, train_grid_ms) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0 and epoch > 0:
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    val_pbar = tqdm(total=len(val_dataset_loader), desc='Validation')
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, val_grid_ms) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_grid_ms_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid_ms]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_label_tensor.shape[0],
                                                  train_vox_ten_ms=val_grid_ms_ten)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels, dim=1).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                        val_pbar.update(1)
                    val_pbar.close()
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                val_iou_dict = {}
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                    val_iou_dict[f'val_iou/{class_name}'] = class_iou * 100
                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                # Log validation metrics to wandb
                wandb.log({
                    'val/miou': val_miou,
                    'val/loss': np.mean(val_loss_list),
                    **val_iou_dict,
                    'epoch': epoch,
                    'global_iter': global_iter,
                })

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                # always save full checkpoint for crash recovery
                save_full_checkpoint(checkpoint_path, my_model, optimizer, epoch, global_iter, best_val_miou)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))

                wandb.log({
                    'val/best_miou': best_val_miou,
                    'global_iter': global_iter,
                })

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            train_vox_ms_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid_ms]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0],
                               train_vox_ten_ms=train_vox_ms_ten)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs, dim=1), point_label_tensor, ignore=0) + loss_func(
                outputs, point_label_tensor)
            loss = loss / accumulation_steps
            loss.backward()

            if (i_iter + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_list.append(loss.item() * accumulation_steps)

            # Log training loss to wandb every step
            wandb.log({
                'train/loss': loss.item(),
                'epoch': epoch,
                'global_iter': global_iter,
            })

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                    wandb.log({
                        'train/avg_loss': np.mean(loss_list),
                        'epoch': epoch,
                        'global_iter': global_iter,
                    })
                else:
                    print('loss error')
        pbar.close()
        # Flush any leftover accumulated gradients at end of epoch
        if len(train_dataset_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        epoch += 1
        # Save checkpoint at end of every epoch for crash recovery
        save_full_checkpoint(checkpoint_path, my_model, optimizer, epoch, global_iter, best_val_miou)

    wandb.finish()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
