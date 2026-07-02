"""
Evaluation script for the distilled NUC-Net student model.

Evaluates the student model on SemanticKITTI validation set and reports:
  - Per-class IoU and mIoU
  - Inference speed (ms/frame), FPS
  - Peak VRAM usage (MB)
  - Parameter count (M)

This script can also compare student vs teacher metrics side-by-side
when a teacher checkpoint is provided.

Usage (run from any directory - all paths are resolved relative to your CWD):
    cd src/NUC-Net/distillation
    python evaluation/eval_student.py -y config/distill_semantickitti.yaml

    # With teacher comparison:
    python evaluation/eval_student.py -y config/distill_semantickitti.yaml --compare_teacher

    # Override student checkpoint path:
    python evaluation/eval_student.py -y config/distill_semantickitti.yaml -m ./checkpoints/student_ss_best.pt -p ./predictions
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (same as training script)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DISTILL_ROOT = _SCRIPT_DIR.parent
_NUCNET_DIR = _DISTILL_ROOT.parent  # NUC-Net/
_NUCNET_ROOT = _NUCNET_DIR / "Cylinder3d_with_NUC"

sys.path.insert(0, str(_DISTILL_ROOT))
sys.path.insert(0, str(_NUCNET_ROOT))
_ORIGINAL_CWD = Path.cwd()  # Save original CWD before changing
torch.cuda.init()
os.chdir(str(_NUCNET_DIR))

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder

from models.teacher import build_teacher
from models.student import build_student

import warnings
warnings.filterwarnings("ignore")


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


import time

class InferenceProfiler:
    """Measures per-frame inference timing with GPU synchronization."""

    def __init__(self, device):
        self.device = device
        self.times = []
        self._start_event = None
        self._end_event = None
        self._start_time = None

    def start_frame(self):
        if self.device.type == 'cuda':
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.time()

    def end_frame(self):
        if self.device.type == 'cuda':
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            elapsed_ms = (time.time() - self._start_time) * 1000.0
        self.times.append(elapsed_ms)

    def get_stats(self):
        # Skip first frame (GPU/CPU warmup)
        times = self.times[1:] if len(self.times) > 1 else self.times
        avg_ms = np.mean(times) if times else 0.0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if self.device.type == 'cuda' else 0.0
        return {
            'inference_speed_ms': avg_ms,
            'fps': fps,
            'vram_peak_mb': vram_mb,
            'total_frames': len(self.times),
        }


def evaluate_model(model, val_loader, unique_label, unique_label_str, device,
                   model_name="Model", pred_dir=None, remap_lut=None,
                   val_pt_dataset=None, use_amp=False):
    """
    Run inference + compute metrics for a single model.

    Returns:
        metrics: dict with mIoU, per-class IoU, speed, VRAM, params
    """
    model.eval()
    profiler = InferenceProfiler(device)
    hist_list = []
    sample_idx = 0

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _, val_vox_label, val_grid, val_pt_labs, val_pt_fea, val_grid_ms in tqdm(
                val_loader, desc=f"Evaluating {model_name}"):

            pt_fea_ten = [torch.from_numpy(i).float().to(device) for i in val_pt_fea]
                
            grid_ten = [torch.from_numpy(i).to(device) for i in val_grid]
            grid_ms_ten = [torch.from_numpy(i).to(device) for i in val_grid_ms]

            profiler.start_frame()
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                outputs = model(pt_fea_ten, grid_ten, val_vox_label.shape[0],
                                train_vox_ten_ms=grid_ms_ten)
            profiler.end_frame()

            # Handle both dict outputs (distill model) and tensor outputs (original)
            if isinstance(outputs, dict):
                logits = outputs['dense_logits']
            else:
                logits = outputs

            predict_labels = torch.argmax(logits, dim=1).cpu().numpy()

            for count, _ in enumerate(val_grid):
                per_point_pred = predict_labels[
                    count, val_grid[count][:, 0],
                    val_grid[count][:, 1],
                    val_grid[count][:, 2]
                ]
                hist_list.append(fast_hist_crop(per_point_pred, val_pt_labs[count],
                                               unique_label))

                # Save per-frame .label file (Alpine-compatible)
                if pred_dir and remap_lut is not None and val_pt_dataset is not None:
                    pred_remapped = remap_lut[per_point_pred].astype(np.uint32)
                    vel_path = val_pt_dataset.im_idx[sample_idx].replace('\\', '/')
                    _, rel = vel_path.rsplit('/sequences/', 1)
                    save_name = rel.replace('velodyne', 'predictions')[:-3] + 'label'
                    save_file = os.path.join(pred_dir, 'sequences', *save_name.split('/'))
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    pred_remapped.tofile(save_file)
                    sample_idx += 1

    iou = per_class_iu(sum(hist_list))
    profiler_stats = profiler.get_stats()

    # Build metrics dict in the same format as NUC-Net's collect_backbone_metrics
    # so Alpine's load_backbone_stats / merge pipeline can consume it directly.
    metrics = {
        'num_params_M': total_params,
        'mIoU': float(np.nanmean(iou)),
    }
    for name, iou_val in zip(unique_label_str, iou):
        metrics[f'iou/{name}'] = float(iou_val)
    metrics['inference_speed_ms'] = profiler_stats['inference_speed_ms']
    metrics['fps'] = profiler_stats['fps']
    metrics['vram_peak_mb'] = profiler_stats['vram_peak_mb']
    metrics['total_frames'] = profiler_stats['total_frames']

    return metrics


def print_metrics(metrics, title=""):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title or 'Model'} Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  mIoU:             {metrics['mIoU']:.4f}")
    print(f"  Inference Speed:  {metrics['inference_speed_ms']:.2f} ms/frame")
    print(f"  FPS:              {metrics['fps']:.2f}")
    print(f"  VRAM Peak:        {metrics['vram_peak_mb']:.1f} MB")
    print(f"  Parameters:       {metrics['num_params_M']:.2f}M")
    iou_keys = [k for k in metrics if k.startswith('iou/')]
    if iou_keys:
        print(f"\n  Per-class IoU:")
        for k in iou_keys:
            cls_name = k.split('/', 1)[1]
            print(f"    {cls_name:20s}: {metrics[k]:.4f}")
    print(f"{'=' * 60}\n")


def print_comparison(teacher_metrics, student_metrics):
    """Print side-by-side comparison of teacher vs student."""
    print(f"\n{'=' * 70}")
    print(f"  Teacher vs Student Comparison")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25s} {'Teacher':>12s} {'Student':>12s} {'Δ':>10s}")
    print(f"  {'-' * 61}")

    t, s = teacher_metrics, student_metrics
    rows = [
        ('mIoU', t['mIoU'], s['mIoU']),
        ('Speed (ms/frame)', t['inference_speed_ms'], s['inference_speed_ms']),
        ('FPS', t['fps'], s['fps']),
        ('VRAM (MB)', t['vram_peak_mb'], s['vram_peak_mb']),
        ('Params (M)', t['num_params_M'], s['num_params_M']),
    ]
    for name, tv, sv in rows:
        delta = sv - tv
        sign = '+' if delta > 0 else ''
        print(f"  {name:<25s} {tv:>12.2f} {sv:>12.2f} {sign}{delta:>9.2f}")

    # Speedup ratio
    if s['inference_speed_ms'] > 0:
        speedup = t['inference_speed_ms'] / s['inference_speed_ms']
        print(f"\n  Speedup: {speedup:.2f}x")

    # Param reduction
    if t['num_params_M'] > 0:
        reduction = (1 - s['num_params_M'] / t['num_params_M']) * 100
        print(f"  Param Reduction: {reduction:.1f}%")

    print(f"{'=' * 70}\n")


def main(args):
    device = torch.device('cpu') if args.quantized else torch.device('cuda:0')
    config = load_config(args.config_path)

    student_config = config['student_params']
    teacher_config = config['teacher_params']
    dataset_config = config['dataset_params']
    train_dl_config = config['train_data_loader']
    val_dl_config = config['val_data_loader']

    grid_size = student_config['output_shape']
    num_scales = student_config['num_scales']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    # ---- Build data loader ----
    _, val_loader = data_builder.build(
        dataset_config, train_dl_config, val_dl_config,
        grid_size=grid_size, num_scales=num_scales,
    )

    # ---- Prediction saving setup (for Alpine pipeline) ----
    pred_dir = args.pred_dir
    remap_lut = None
    val_pt_dataset = None
    if pred_dir:
        with open(dataset_config["label_mapping"], 'r') as stream:
            label_yaml = yaml.safe_load(stream)
        remapdict = dict(label_yaml['learning_map_inv'])
        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        val_pt_dataset = val_loader.dataset.point_cloud_dataset
        print(f"Will save predictions to: {pred_dir}")

    # ---- Build and load student model ----
    student_model = build_student(student_config, device=str(device))
    if args.quantized:
        student_model = torch.quantization.quantize_dynamic(
            student_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    # Resolve student checkpoint relative to distillation root
    if args.model_path:
        student_ckpt = args.model_path
    else:
        student_ckpt = config['train_params']['model_save_path']
    if os.path.exists(student_ckpt):
        state_dict = torch.load(student_ckpt, map_location=device)
        if isinstance(state_dict, dict) and 'student_state_dict' in state_dict:
            state_dict = state_dict['student_state_dict']
        student_model.load_state_dict(state_dict, strict=False)
        print(f"[Student] Loaded from {student_ckpt}")
    else:
        print(f"[Student] WARNING: No checkpoint at {student_ckpt}, evaluating random init")

    if args.quantized:
        # Move the 3D spconv backbone to GPU to leverage fast sparse operations
        # while keeping the quantized linear layers on CPU
        student_model.cylinder_3d_spconv_seg = student_model.cylinder_3d_spconv_seg.cuda()

    # ---- Evaluate student ----
    student_metrics = evaluate_model(student_model, val_loader, unique_label,
                                     unique_label_str, device, "Student",
                                     pred_dir=pred_dir, remap_lut=remap_lut,
                                     val_pt_dataset=val_pt_dataset,
                                     use_amp=args.amp)
    print_metrics(student_metrics, "Distilled Student")
    if pred_dir:
        print(f"Predictions saved to: {pred_dir}")

    # ---- Optionally evaluate teacher for comparison ----
    if args.compare_teacher:
        teacher_model = build_teacher(teacher_config, device=str(device))
        teacher_metrics = evaluate_model(teacher_model, val_loader, unique_label,
                                         unique_label_str, device, "Teacher",
                                         use_amp=args.amp)
        print_metrics(teacher_metrics, "Teacher (Full-size)")
        print_comparison(teacher_metrics, student_metrics)

    # ---- Save metrics to JSON (flat format, same as NUC-Net eval) ----
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, 'w') as f:
        json.dump(student_metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NUC-Net Distilled Student Evaluation (SemanticKITTI)')
    parser.add_argument('-y', '--config_path',
                        default='config/distill_semantickitti.yaml',
                        help='Path to distillation config YAML')
    parser.add_argument('-m', '--model_path', default=None,
                        help='Path to student checkpoint (overrides config)')
    parser.add_argument('-o', '--output_dir', default='logs',
                        help='Directory to save metrics JSON')
    parser.add_argument('--compare_teacher', action='store_true',
                        help='Also evaluate teacher model for comparison')
    parser.add_argument('-p', '--pred_dir', default=None,
                        help='Directory to save per-frame .label predictions (Alpine-compatible)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable Mixed Precision (autocast) for inference')
    parser.add_argument('--quantized', action='store_true',
                        help='Evaluate INT8 dynamically quantized model on CPU')
    args = parser.parse_args()

    # Resolve paths relative to the original CWD (before os.chdir)
    args.config_path = str((_ORIGINAL_CWD / args.config_path).resolve())
    if args.model_path:
        args.model_path = str((_ORIGINAL_CWD / args.model_path).resolve())

    args.output_dir = str((_ORIGINAL_CWD / args.output_dir).resolve())
    if args.pred_dir:
        args.pred_dir = str((_ORIGINAL_CWD / args.pred_dir).resolve())

    print(' '.join(sys.argv))
    print(args)
    main(args)
