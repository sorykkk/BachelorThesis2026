"""
Teacher Model: loads the full-size pretrained NUC-Net and freezes all weights.

The teacher's role in distillation is to provide:
  1. Soft logit targets   (for KL divergence loss)
  2. Feature targets      (for feature MSE loss)
  3. Boundary affinities  (for local consistency loss)

All parameters are frozen (requires_grad=False) so the teacher
is never updated during distillation training.
"""

import os
import torch
from torch import nn
from pathlib import Path

from network.cylinder_fea_generator import cylinder_fea
from models.feature_extractor import Asymm3DSpconvDistill, CylinderAsymDistill

# NUC-Net/ directory (parent of distillation/)
_NUCNET_DIR = Path(__file__).resolve().parents[2]


def build_teacher(teacher_config, device='cuda:0'):
    """
    Build and load the pretrained teacher model.

    Args:
        teacher_config: dict with keys matching model_params in config YAML
            Required: output_shape, num_class, num_input_features, use_norm,
                      init_size, fea_dim, out_fea_dim, num_scales, checkpoint_path
        device: torch device string

    Returns:
        teacher_model: CylinderAsymDistill with frozen weights, in eval mode
    """
    output_shape = teacher_config['output_shape']
    num_class = teacher_config['num_class']
    num_input_features = teacher_config['num_input_features']
    use_norm = teacher_config['use_norm']
    init_size = teacher_config['init_size']
    fea_dim = teacher_config['fea_dim']
    out_fea_dim = teacher_config['out_fea_dim']
    num_scales = teacher_config['num_scales']
    # Resolve checkpoint path relative to NUC-Net/ directory
    checkpoint_path = str(_NUCNET_DIR / teacher_config['checkpoint_path'])

    # With NUMA, the backbone input dimension is num_input_features * num_scales
    # because the feature generator concatenates features from each scale
    spconv_input_features = num_input_features * num_scales

    # Build the distillation-aware backbone (same architecture as teacher, returns features)
    backbone = Asymm3DSpconvDistill(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=spconv_input_features,
        init_size=init_size,
        nclasses=num_class,
    )

    # Build the cylinder feature generator (MLP + NUMA multi-scale pooling)
    fea_generator = cylinder_fea(
        grid_size=output_shape,
        fea_dim=fea_dim,
        out_pt_fea_dim=out_fea_dim,
        fea_compre=num_input_features,
        num_scales=num_scales,
    )

    # Wrap into the full model
    teacher_model = CylinderAsymDistill(
        cylin_model=fea_generator,
        segmentator_spconv=backbone,
        sparse_shape=output_shape,
    )

    # Load pretrained weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle both full-checkpoint format and raw state_dict format
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # The teacher uses CylinderAsymDistill which has same parameter names
        # as the original cylinder_asym, so weights should load directly
        teacher_model.load_state_dict(state_dict, strict=False)
        print(f"[Teacher] Loaded pretrained weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"[Teacher] Checkpoint not found at {checkpoint_path}. "
            "Train the full NUC-Net first or provide a valid checkpoint path."
        )

    # Move to device and freeze all parameters
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Count parameters (for logging)
    total_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
    print(f"[Teacher] Total params: {total_params:.2f}M (all frozen)")

    return teacher_model
