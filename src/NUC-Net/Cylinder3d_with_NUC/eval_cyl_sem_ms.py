# -*- coding:utf-8 -*-
# Inference-time evaluation for NUC-Net (Cylinder3D with NUC)
# Collects: Inference Speed [ms/frame], FPS [frames/s], VRAM peak [MB],
#           mIoU, Num Params [M], FLOPS [G, param_estimate]

# USAGE:
# cd src/NUC-Net/Cylinder3d_with_NUC
# # Default (uses config's model_load_path)
# python eval_cyl_sem_ms.py -y config/semantickitti-multiscan.yaml
# # Override checkpoint path
# python eval_cyl_sem_ms.py -y config/semantickitti-multiscan.yaml -m ./model_save_dir/model_full_ms.pt
# # Custom output dir
# python eval_cyl_sem_ms.py -y config/semantickitti-multiscan.yaml -o ./performance_logs

import os
import sys
import argparse
import time
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
from evaluation.metrics_profiler import (
    InferenceProfiler,
    count_parameters,
    collect_backbone_metrics,
    print_backbone_metrics_table,
    save_metrics_json,
)

import warnings
warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    configs = load_config_data(args.config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = args.model_path if args.model_path else train_hypers.get('model_load_path', train_hypers['model_save_path'])

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    # ---- Build model ----
    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)
        print(f"Loaded checkpoint from {model_load_path}")
    else:
        print(f"WARNING: checkpoint not found at {model_load_path}")

    my_model.to(pytorch_device)
    my_model.eval()

    # ---- Model complexity metrics (param count) ----
    model_stats = count_parameters(my_model)
    print(f"Num Params: {model_stats['total_params_M']:.2f}M  "
          f"(trainable: {model_stats['trainable_params_M']:.2f}M)")

    # ---- Build validation dataloader ----
    use_multiscan = 'multiscan' in args.config_path
    train_dataset_loader, val_dataset_loader = data_builder.build(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        grid_size=grid_size,
        use_multiscan=use_multiscan,
    )

    # ---- Prediction saving setup (for Alpine pipeline) ----
    if args.pred_dir:
        with open(dataset_config["label_mapping"], 'r') as stream:
            label_yaml = yaml.safe_load(stream)
        remapdict = dict(label_yaml['learning_map_inv'])
        # For multiscan: add inverse entries for moving classes not in learning_map_inv
        for orig_label, mapped_label in label_yaml.get('learning_map', {}).items():
            if mapped_label not in remapdict:
                remapdict[mapped_label] = orig_label
        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        val_pt_dataset = val_dataset_loader.dataset.point_cloud_dataset
        sample_idx = 0
        print(f"Will save predictions to: {args.pred_dir}")

    # ---- Inference loop with profiling ----
    profiler = InferenceProfiler()
    hist_list = []

    print(f"\nRunning inference on {len(val_dataset_loader)} batches...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i_iter, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, origin_len) in enumerate(
                tqdm(val_dataset_loader, desc="Inference")):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device)
                              for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]

            # --- Timed forward pass ---
            profiler.start_frame()
            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            profiler.end_frame()

            # --- Compute per-class IoU and optionally save predictions ---
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_val_grid in enumerate(val_grid):
                per_point_pred = predict_labels[
                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                    val_grid[count][:, 2]][0:origin_len[count]]
                hist_list.append(fast_hist_crop(
                    per_point_pred,
                    val_pt_labs[count][0:origin_len[count]],
                    unique_label))

                # Save per-frame .label file (Alpine-compatible)
                if args.pred_dir:
                    pred_remapped = remap_lut[per_point_pred].astype(np.uint32)
                    vel_path = val_pt_dataset.im_idx[sample_idx].replace('\\', '/')
                    _, rel = vel_path.rsplit('/sequences/', 1)
                    save_name = rel.replace('velodyne', 'predictions')[:-3] + 'label'
                    save_file = os.path.join(args.pred_dir, 'sequences', *save_name.split('/'))
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    pred_remapped.tofile(save_file)
                    sample_idx += 1

    # ---- Compute final metrics ----
    iou = per_class_iu(sum(hist_list))
    val_miou = float(np.nanmean(iou))

    profiler_stats = profiler.get_stats()

    all_metrics = collect_backbone_metrics(
        profiler_stats=profiler_stats,
        model_stats=model_stats,
        miou=val_miou,
        per_class_iou=iou,
        class_names=unique_label_str,
    )
    all_metrics["batch_size"] = val_batch_size

    # ---- Output ----
    dataset_name = "SemanticKITTI-multiscan" if use_multiscan else "SemanticKITTI"
    print_backbone_metrics_table(all_metrics, dataset_name=dataset_name, model_name="NUC-Net")

    os.makedirs(args.output_dir, exist_ok=True)
    save_metrics_json(all_metrics, os.path.join(args.output_dir, "metrics.json"))

    if args.pred_dir:
        print(f"Predictions saved to: {args.pred_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NUC-Net inference evaluation with metrics profiling')
    parser.add_argument('-y', '--config_path', default='config/semantickitti-multiscan.yaml',
                        help='Path to config YAML')
    parser.add_argument('-m', '--model_path', default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('-o', '--output_dir', default='./logs',
                        help='Directory to save metrics JSON')
    parser.add_argument('-p', '--pred_dir', default=None,
                        help='Directory to save per-frame .label predictions (Alpine-compatible)')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
