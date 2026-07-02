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

    model_load_path = train_hypers.get('model_load_path', '')
    model_save_path = train_hypers['model_save_path']
    checkpoint_path = train_hypers.get('checkpoint_path', model_save_path.replace('.pt', '_checkpoint.pt'))

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

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_multiscan=True)

    # Initialize wandb
    wandb.init(
        entity='soryn-besleaga-universitatea-politehnica-timi',
        project='NUC-Net-SemanticKITTI',
        config={
            'architecture': 'Cylinder3D_NUC',
            'dataset': 'SemanticKITTI-multiscan',
            'config_file': config_path,
            'learning_rate': train_hypers['learning_rate'],
            'max_epochs': train_hypers['max_num_epochs'],
            'eval_every_n_steps': train_hypers['eval_every_n_steps'],
            'train_batch_size': train_batch_size,
            'val_batch_size': val_batch_size,
            'grid_size': grid_size,
            'num_class': num_class,
        },
        name=f'train-semantickitti-multiscan',
        resume='allow',
    )

    # training
    epoch = resume_epoch
    best_val_miou = resume_best_val_miou
    my_model.train()
    global_iter = resume_global_iter
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea, origin_len) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0 and epoch > 0: # and global_iter != 0:
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    val_pbar = tqdm(total=len(val_dataset_loader), desc='Validation')
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, origin_len) in enumerate(
                            val_dataset_loader):

                        #print(i_iter_val)
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_label_tensor.shape[0])#val_batch_size)
                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]][0:origin_len[count]], val_pt_labs[count][0:origin_len[count]],
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

            #return 0
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0] )#train_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(
                outputs, point_label_tensor)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

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

            optimizer.zero_grad()
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
        epoch += 1
        # Save checkpoint at end of every epoch for crash recovery
        save_full_checkpoint(checkpoint_path, my_model, optimizer, epoch, global_iter, best_val_miou)

    wandb.finish()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti-multiscan.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
