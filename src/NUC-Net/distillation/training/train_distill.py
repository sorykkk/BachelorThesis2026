"""
NUC-Net Knowledge Distillation Training Script (SemanticKITTI, single-scan).

This script trains a smaller (50% width) student model to match a
pretrained full-size teacher model using multi-component distillation:

    L_total = L_GT + lambda_1 * L_feat + lambda_2 * L_KL + lambda_3 * L_boundary

Components:
    L_GT       - Cross-Entropy + Lovász-Softmax with ground truth labels
    L_feat     - MSE between adapted student features and teacher features
    L_KL       - KL divergence on temperature-softened logits
    L_boundary - Local boundary affinity matching

Both teacher and student use the same NUC partitioning and NUMA
multi-scale aggregation - only the 3D backbone channel widths differ.

Usage:
    cd src/NUC-Net/distillation
    python training/train_distill.py -y config/distill_semantickitti.yaml

    # Resume from checkpoint:
    python training/train_distill.py -y config/distill_semantickitti.yaml --resume
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: add the original NUC-Net codebase to Python path so we can
# import its data loaders, datasets, losses (Lovász), and utilities.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DISTILL_ROOT = _SCRIPT_DIR.parent                          # distillation/
_NUCNET_ROOT = _DISTILL_ROOT.parent / "Cylinder3d_with_NUC" # Cylinder3d_with_NUC/

# Add both directories to sys.path for imports
sys.path.insert(0, str(_DISTILL_ROOT))    # for models/, losses/, training/
sys.path.insert(0, str(_NUCNET_ROOT))     # for network/, builder/, dataloader/, utils/

# Change working directory to NUC-Net/ so all config paths
# (data, label_mapping, checkpoints) resolve relative to this directory
_NUCNET_DIR = _DISTILL_ROOT.parent  # NUC-Net/
_ORIGINAL_CWD = Path.cwd()  # Save original CWD before changing
os.chdir(str(_NUCNET_DIR))

import wandb

# Auto-load WANDB_API_KEY from .env.local (same mechanism as original training)
# Use _SCRIPT_DIR (resolved before os.chdir) to avoid CWD-dependent resolution
_env_file = _SCRIPT_DIR.parents[2] / ".env.local"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip().strip('"'))
    print(f"[wandb] Loaded env from {_env_file}")
if "WANDB_API_KEY" in os.environ:
    os.environ["WANDB_MODE"] = "online"
else:
    print("[wandb] WARNING: WANDB_API_KEY not set. Running in offline mode.")
    os.environ["WANDB_MODE"] = "offline"

# Imports from original NUC-Net codebase
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder

# Imports from distillation package
from models.teacher import build_teacher
from models.student import build_student
from models.adaptation import build_multi_scale_adaptation
from losses.combined_loss import CombinedDistillLoss

import warnings
warnings.filterwarnings("ignore")


def load_config(config_path):
    """Load YAML config using strictyaml-compatible loader."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(path, student_model, adaptation_layer, optimizer, epoch,
                    global_iter, best_val_miou):
    """Save full training checkpoint for crash recovery."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'student_state_dict': student_model.state_dict(),
        'adaptation_state_dict': adaptation_layer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_iter': global_iter,
        'best_val_miou': best_val_miou,
    }, path)
    print(f"[checkpoint] Saved to {path} (epoch={epoch}, iter={global_iter}, "
          f"best_miou={best_val_miou:.3f})")


def load_checkpoint(path, student_model, adaptation_layer, optimizer=None,
                    device='cuda:0'):
    """Load training checkpoint. Returns (model, adapt, optimizer, epoch, iter, miou)."""
    checkpoint = torch.load(path, map_location=device)
    student_model.load_state_dict(checkpoint['student_state_dict'])
    adaptation_layer.load_state_dict(checkpoint['adaptation_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    global_iter = checkpoint.get('global_iter', 0)
    best_val_miou = checkpoint.get('best_val_miou', 0.0)
    print(f"[checkpoint] Resumed from {path} (epoch={epoch}, iter={global_iter}, "
          f"best_miou={best_val_miou:.3f})")
    return student_model, adaptation_layer, optimizer, epoch, global_iter, best_val_miou


def validate(student_model, val_loader, unique_label, unique_label_str, device, use_amp=False):
    """Run validation and compute per-class IoU / mIoU."""
    student_model.eval()
    hist_list = []

    with torch.no_grad():
        for _, val_vox_label, val_grid, val_pt_labs, val_pt_fea, val_grid_ms in tqdm(
                val_loader, desc="Validation"):

            val_pt_fea_ten = [torch.from_numpy(i).float().to(device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(device) for i in val_grid]
            val_grid_ms_ten = [torch.from_numpy(i).to(device) for i in val_grid_ms]

            # Student forward pass (returns dict)
            outputs = student_model(val_pt_fea_ten, val_grid_ten,
                                    val_vox_label.shape[0],
                                    train_vox_ten_ms=val_grid_ms_ten)

            predict_labels = torch.argmax(outputs['dense_logits'], dim=1)
            predict_labels = predict_labels.cpu().numpy()

            for count, _ in enumerate(val_grid):
                per_point_pred = predict_labels[
                    count, val_grid[count][:, 0],
                    val_grid[count][:, 1],
                    val_grid[count][:, 2]
                ]
                hist_list.append(fast_hist_crop(per_point_pred, val_pt_labs[count],
                                               unique_label))

    iou = per_class_iu(sum(hist_list))
    val_miou = np.nanmean(iou) * 100

    # Print per-class IoU
    iou_dict = {}
    for class_name, class_iou in zip(unique_label_str, iou):
        iou_dict[f'val_iou/{class_name}'] = class_iou * 100

    return val_miou, iou_dict


def main(args):
    device = torch.device('cuda:0')
    config = load_config(args.config_path)

    # ---- Unpack config sections ----
    teacher_config = config['teacher_params']
    student_config = config['student_params']
    dataset_config = config['dataset_params']
    train_dl_config = config['train_data_loader']
    val_dl_config = config['val_data_loader']
    distill_config = config['distill_params']
    train_config = config['train_params']

    grid_size = student_config['output_shape']
    num_class = student_config['num_class']
    ignore_label = dataset_config['ignore_label']
    num_scales = student_config['num_scales']

    # ---- Label names for logging ----
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    # ==================================================================
    #  BUILD MODELS
    # ==================================================================

    # ---- Teacher: frozen pretrained model ----
    print("\n" + "=" * 60)
    print("Building TEACHER model (frozen)")
    print("=" * 60)
    teacher_model = build_teacher(teacher_config, device=str(device))

    # ---- Student: reduced-width model (trainable) ----
    print("\n" + "=" * 60)
    print("Building STUDENT model (trainable)")
    print("=" * 60)
    student_model = build_student(student_config, device=str(device))

    # ---- Adaptation layer: multi-scale (student features -> teacher space) ----
    adaptation_layer = build_multi_scale_adaptation(teacher_config, student_config,
                                                    device=str(device))

    # ==================================================================
    #  OPTIMIZER & SCHEDULER
    # ==================================================================

    # Optimize both student model parameters and adaptation layer parameters
    all_params = list(student_model.parameters()) + list(adaptation_layer.parameters())
    optimizer = optim.Adam(all_params, lr=train_config['learning_rate'])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ==================================================================
    #  LOSS FUNCTION
    # ==================================================================

    distill_loss = CombinedDistillLoss(distill_config, num_class=num_class,
                                       ignore_label=ignore_label)

    # ==================================================================
    #  RESUME FROM CHECKPOINT
    # ==================================================================

    start_epoch = 0
    global_iter = 0
    best_val_miou = 0.0

    # Checkpoint paths are relative to NUC-Net/ (CWD)
    checkpoint_path = train_config['checkpoint_path']
    model_save_path_resolved = train_config['model_save_path']

    if args.resume and os.path.exists(checkpoint_path):
        (student_model, adaptation_layer, optimizer,
         start_epoch, global_iter, best_val_miou) = load_checkpoint(
            checkpoint_path, student_model, adaptation_layer, optimizer,
            device=str(device))

    # ==================================================================
    #  INSTANCE AUGMENTATION (IA)
    # ==================================================================

    instance_augmentor = None
    ia_db_path = train_config.get('instance_db_path', '')
    if ia_db_path and os.path.exists(ia_db_path):
        from dataloader.instance_augmentation import InstanceAugmentor
        instance_augmentor = InstanceAugmentor(ia_db_path, max_paste_per_class=3)
        print(f"[IA] Instance Augmentation ENABLED from {ia_db_path}")
    else:
        print(f"[IA] Instance Augmentation DISABLED (db not found at '{ia_db_path}')")

    # ==================================================================
    #  DATA LOADERS
    # ==================================================================

    print("\n" + "=" * 60)
    print("Building data loaders")
    print("=" * 60)
    train_loader, val_loader = data_builder.build(
        dataset_config, train_dl_config, val_dl_config,
        grid_size=grid_size, instance_augmentor=instance_augmentor,
        num_scales=num_scales,
    )

    # Track epoch length for instance augmentor logging
    if instance_augmentor is not None:
        instance_augmentor._epoch_length = len(train_loader.dataset)

    # ==================================================================
    #  WANDB LOGGING
    # ==================================================================

    wandb.init(
        entity='soryn-besleaga-universitatea-politehnica-timi',
        project='NUC-Net-Distillation',
        config={
            'architecture': 'NUC-Net-Distill',
            'dataset': 'SemanticKITTI',
            'teacher_init_size': teacher_config['init_size'],
            'student_init_size': student_config['init_size'],
            'temperature': distill_config['temperature'],
            'lambda_feat_init': distill_config['lambda_feat'],
            'lambda_kl_init': distill_config['lambda_kl'],
            'lambda_boundary_init': distill_config['lambda_boundary'],
            'warmup_epochs': distill_config['warmup_epochs'],
            'learning_rate': train_config['learning_rate'],
            'max_epochs': train_config['max_num_epochs'],
            'batch_size': train_dl_config['batch_size'],
            'accumulation_steps': train_config['accumulation_steps'],
            # Distillation hyperparams
            'multi_scale_weights': distill_config.get('multi_scale_weights', [0.5, 1.0, 2.0, 4.0]),
            'cosine_weight': distill_config.get('cosine_weight', 0.5),
            'use_decoupled_kd': distill_config.get('use_decoupled_kd', True),
            'dkd_alpha': distill_config.get('dkd_alpha', 1.0),
            'dkd_beta': distill_config.get('dkd_beta', 8.0),
            'use_logit_standardization': distill_config.get('use_logit_standardization', True),
            'use_entropy_weighting': distill_config.get('use_entropy_weighting', True),
            'instance_augmentation': instance_augmentor is not None,
        },
        name='distill-semantickitti-ss',
        resume='allow',
    )

    # ==================================================================
    #  TRAINING LOOP
    # ==================================================================

    accumulation_steps = train_config['accumulation_steps']
    check_iter = train_config['eval_every_n_steps']
    model_save_path = model_save_path_resolved

    print(f"\n{'=' * 60}")
    print(f"Distillation Training Config")
    print(f"{'=' * 60}")
    print(f"  Teacher init_size: {teacher_config['init_size']}")
    print(f"  Student init_size: {student_config['init_size']}")
    print(f"  Temperature (τ):   {distill_config['temperature']}")
    print(f"  Batch size:        {train_dl_config['batch_size']} x {accumulation_steps} = "
          f"{train_dl_config['batch_size'] * accumulation_steps} effective")
    print(f"  Learning rate:     {train_config['learning_rate']}")
    print(f"  Max epochs:        {train_config['max_num_epochs']}")
    print(f"  Lambda schedule:   {distill_loss.lambda_scheduler}")
    print(f"{'=' * 60}\n")

    epoch = start_epoch
    student_model.train()

    while epoch < train_config['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        time.sleep(2)
        optimizer.zero_grad()

        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea, train_grid_ms) in enumerate(train_loader):

            # ---- Periodic validation ----
            if global_iter % check_iter == 0 and epoch > 0:
                val_miou, val_iou_dict = validate(
                    student_model, val_loader, unique_label, unique_label_str, device, use_amp=args.amp)

                if np.isnan(val_miou):
                    print("\n[FATAL ERROR] Validation mIoU is NaN. The model has likely collapsed (e.g., due to FP16 overflow or exploding gradients).")
                    print("Interrupting training immediately to prevent overwriting the good checkpoint with corrupted weights.")
                    sys.exit(1)

                print(f"\n[Validation] mIoU: {val_miou:.2f}% (best: {best_val_miou:.2f}%)")
                for name, iou_val in val_iou_dict.items():
                    print(f"  {name}: {iou_val:.2f}%")

                wandb.log({
                    'val/miou': val_miou,
                    **val_iou_dict,
                    'epoch': epoch,
                    'global_iter': global_iter,
                })

                # Save best model
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(student_model.state_dict(), model_save_path)
                    print(f"[BEST] New best mIoU: {best_val_miou:.2f}% -> saved to {model_save_path}")

                # Always save checkpoint for crash recovery
                save_checkpoint(checkpoint_path, student_model, adaptation_layer,
                                optimizer, epoch, global_iter, best_val_miou)

                wandb.log({'val/best_miou': best_val_miou, 'global_iter': global_iter})
                student_model.train()

            # ---- Prepare input tensors ----
            pt_fea_ten = [torch.from_numpy(i).float().to(device) for i in train_pt_fea]
            grid_ten = [torch.from_numpy(i).to(device) for i in train_grid]
            grid_ms_ten = [torch.from_numpy(i).to(device) for i in train_grid_ms]
            label_tensor = train_vox_label.type(torch.LongTensor).to(device)
            batch_size = label_tensor.shape[0]

            # ---- Teacher forward (no grad, frozen) ----
            # Keep teacher in FP32 as its backbone might have issues with spconv FP16 kernels
            with torch.no_grad():
                teacher_outputs = teacher_model(pt_fea_ten, grid_ten, batch_size,
                                                train_vox_ten_ms=grid_ms_ten)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # ---- Student forward (with grad, trainable) ----
                student_outputs = student_model(pt_fea_ten, grid_ten, batch_size,
                                                train_vox_ten_ms=grid_ms_ten)

            # Move loss computation outside of autocast to prevent overflow in distance metrics
            with torch.cuda.amp.autocast(enabled=False):
                # Ensure outputs are FP32
                if args.amp:
                    for k, v in student_outputs.items():
                        if isinstance(v, torch.Tensor):
                            student_outputs[k] = v.float()
                        elif isinstance(v, dict):
                            student_outputs[k] = {k2: v2.float() for k2, v2 in v.items()}
                
                # ---- Combined distillation loss ----
                loss, loss_dict = distill_loss(
                    student_outputs, teacher_outputs, label_tensor,
                    adaptation_layer, epoch)

                # ---- Gradient accumulation ----
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i_iter + 1) % accumulation_steps == 0:
                # Unscale gradients before clipping to prevent NaN weights
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_list.append(loss.item() * accumulation_steps)

            # ---- Logging ----
            log_dict = {
                'train/loss_total': loss_dict['loss_total'],
                'train/loss_gt': loss_dict['loss_gt'],
                'train/loss_feat': loss_dict['loss_feat'],
                'train/loss_kl': loss_dict['loss_kl'],
                'train/loss_boundary': loss_dict['loss_boundary'],
                'train/lambda_feat': loss_dict['lambda_feat'],
                'train/lambda_kl': loss_dict['lambda_kl'],
                'train/lambda_boundary': loss_dict['lambda_boundary'],
                'epoch': epoch,
                'global_iter': global_iter,
            }
            # Log per-level feature losses (multi-scale breakdown)
            for key, val in loss_dict.items():
                if key.startswith('feat_'):
                    log_dict[f'train/{key}'] = val
            wandb.log(log_dict)

            if global_iter % 500 == 0 and len(loss_list) > 0:
                print(f"  epoch {epoch} iter {i_iter:5d} | "
                      f"total={np.mean(loss_list):.4f} "
                      f"gt={loss_dict['loss_gt']:.4f} "
                      f"feat={loss_dict['loss_feat']:.4f} "
                      f"kl={loss_dict['loss_kl']:.4f} "
                      f"bound={loss_dict['loss_boundary']:.4f} | "
                      f"lambda_feat={loss_dict['lambda_feat']:.2f} "
                      f"lambda_kl={loss_dict['lambda_kl']:.2f} "
                      f"lambda_boundary={loss_dict['lambda_boundary']:.2f}")

            pbar.update(1)
            global_iter += 1

        pbar.close()

        # Flush any remaining accumulated gradients at epoch end
        if len(train_loader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch += 1

        # Save checkpoint at end of every epoch
        save_checkpoint(checkpoint_path, student_model, adaptation_layer,
                        optimizer, epoch, global_iter, best_val_miou)

    # ---- Final validation ----
    val_miou, val_iou_dict = validate(
        student_model, val_loader, unique_label, unique_label_str, device, use_amp=args.amp)
        
    if np.isnan(val_miou):
        print("\n[FATAL ERROR] Final validation mIoU is NaN. The model has likely collapsed.")
        sys.exit(1)
        
    print(f"\n[Final] mIoU: {val_miou:.2f}% (best: {best_val_miou:.2f}%)")

    wandb.finish()
    print("\nDistillation training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NUC-Net Knowledge Distillation Training (SemanticKITTI)')
    parser.add_argument('-y', '--config_path',
                        default='config/distill_semantickitti.yaml',
                        help='Path to distillation config YAML')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--amp', action='store_true',
                        help='Enable Automatic Mixed Precision (AMP)')
    args = parser.parse_args()

    # Resolve config path relative to the original CWD (before os.chdir)
    args.config_path = str((_ORIGINAL_CWD / args.config_path).resolve())

    print(' '.join(sys.argv))
    print(args)
    main(args)
