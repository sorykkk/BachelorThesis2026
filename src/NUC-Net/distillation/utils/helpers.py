"""
Utility / helper functions for NUC-Net distillation.
"""

import torch
import numpy as np


def count_parameters(model):
    """
    Count total and trainable parameters of a model.

    Returns:
        dict with 'total_params_M' and 'trainable_params_M'
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params_M': total / 1e6,
        'trainable_params_M': trainable / 1e6,
    }


def print_model_summary(model, name="Model"):
    """Print a summary of model parameters by module."""
    print(f"\n{name} Parameter Summary:")
    print(f"{'Module':<50s} {'Params':>12s}")
    print("-" * 64)
    for module_name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {module_name:<48s} {n_params / 1e6:>10.3f}M")
    total = sum(p.numel() for p in model.parameters())
    print("-" * 64)
    print(f"  {'TOTAL':<48s} {total / 1e6:>10.3f}M")


def compute_flops_estimate(teacher_config, student_config):
    """
    Rough FLOPs estimate based on channel widths.

    For sparse convolutions, FLOPs scale approximately with
    (in_channels x out_channels x kernel_size^3 x num_active_voxels).
    Since only channels change (not spatial), the ratio is:
        student_flops / teacher_flops ~ (student_init / teacher_init)^2

    This is a simplification - actual FLOPs depend on sparsity patterns.

    Returns:
        dict with flops_ratio and estimated_reduction
    """
    teacher_init = teacher_config['init_size']
    student_init = student_config['init_size']

    # Channel ratio (applies to each conv layer)
    ratio = (student_init / teacher_init) ** 2

    return {
        'flops_ratio': ratio,
        'estimated_reduction_pct': (1 - ratio) * 100,
        'teacher_init_size': teacher_init,
        'student_init_size': student_init,
    }
