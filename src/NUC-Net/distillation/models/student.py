"""
Student Model: builds a reduced-width NUC-Net for distillation training.

Key design decisions (from NUC-Net Distillation Strategy):
  - Width reduction, NOT depth reduction: all layers are kept, but the
    channel count in every 3D sparse convolution is halved.
  - Same NUC partition parameters (a0=0.05, d=0.0062) are preserved.
  - Same NUMA multi-scale aggregation (num_scales=2) is preserved.
  - The spatial resolution (output_shape=[120, 360, 32]) is unchanged.

Teacher vs student channel comparison:
  Component           | Teacher (init_size=32)        | Student (init_size=16)
  --------------------|-------------------------------|------------------------
  Backbone input      | 64 (32 x 2 scales)            | 32 (16 x 2 scales)
  ResBlock encoder    | 32 -> 64 -> 128 -> 256 -> 512 | 16 -> 32 -> 64 -> 128 -> 256
  Decoder             | 512 -> 256 -> 128 -> 64       | 256 -> 128 -> 64 -> 32
  Pre-logit features  | 128 (4 x 32)                  | 64 (4 x 16)
  Logit output        | 20 classes                    | 20 classes
"""

import torch
from torch import nn

from network.cylinder_fea_generator import cylinder_fea
from models.feature_extractor import Asymm3DSpconvDistill, CylinderAsymDistill


def build_student(student_config, device='cuda:0'):
    """
    Build the student model with reduced channel widths.

    Args:
        student_config: dict with keys matching model_params in config YAML
            Required: output_shape, num_class, num_input_features, use_norm,
                      init_size, fea_dim, out_fea_dim, num_scales
        device: torch device string

    Returns:
        student_model: CylinderAsymDistill with trainable weights
    """
    output_shape = student_config['output_shape']
    num_class = student_config['num_class']
    num_input_features = student_config['num_input_features']   # 16 (half of teacher's 32)
    use_norm = student_config['use_norm']
    init_size = student_config['init_size']                     # 16 (half of teacher's 32)
    fea_dim = student_config['fea_dim']
    out_fea_dim = student_config['out_fea_dim']
    num_scales = student_config['num_scales']

    # NUMA: backbone input dim = per-scale features x number of scales
    spconv_input_features = num_input_features * num_scales

    # Build the distillation-aware backbone (halved channels)
    backbone = Asymm3DSpconvDistill(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=spconv_input_features,    # 16*2 = 32 (vs teacher's 64)
        init_size=init_size,                         # 16 (vs teacher's 32)
        nclasses=num_class,
    )

    # Build the cylinder feature generator
    # MLP is kept at full size (relatively cheap compared to 3D convolutions)
    # Only the per-scale feature compression is halved (fea_compre=16 vs 32)
    fea_generator = cylinder_fea(
        grid_size=output_shape,
        fea_dim=fea_dim,
        out_pt_fea_dim=out_fea_dim,                  # 256 (same MLP as teacher)
        fea_compre=num_input_features,               # 16 (half of teacher's 32)
        num_scales=num_scales,
    )

    # Wrap into the full model
    student_model = CylinderAsymDistill(
        cylin_model=fea_generator,
        segmentator_spconv=backbone,
        sparse_shape=output_shape,
    )

    student_model.to(device)

    # Count parameters (for logging)
    total_params = sum(p.numel() for p in student_model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6
    print(f"[Student] Total params: {total_params:.2f}M (trainable: {trainable_params:.2f}M)")

    return student_model
