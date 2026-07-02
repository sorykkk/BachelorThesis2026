"""
Adaptation Layers: map student features to teacher feature space.

Single-level (not used):
    φ: Linear(student_dim -> teacher_dim) + BN

Multi-scale (PointDistiller / BEVDistill):
    φ_l for each backbone stage l ∈ {bottleneck, mid_decoder, late_decoder, pre_logit}

    L_feat = Σ_l α_l · (1/N_l) · Σ_i ||φ_l(F_S^(l,i)) - F_T^(l,i)||^2

No ReLU activation - teacher features from LeakyReLU can be negative,
so the projection must preserve negative values.
"""

import torch
from torch import nn


# Multi-scale level definitions.
# Each level maps to a feature-channel multiplier relative to init_size.
MULTI_SCALE_LEVELS = ['bottleneck', 'mid_decoder', 'late_decoder', 'pre_logit']
LEVEL_MULTIPLIERS = {
    'bottleneck':   16,   # down4c features = 16 * init_size
    'mid_decoder':   8,   # up3e   features =  8 * init_size
    'late_decoder':  2,   # up1e   features =  2 * init_size
    'pre_logit':     4,   # cat(up0e,up1e)  =  4 * init_size
}


class FeatureAdaptation(nn.Module):
    """
    Linear adaptation layer: projects student features into teacher space.

    Architecture: Linear + BN (no ReLU - teacher features can be negative).

    Args:
        student_feat_dim: input dimension  (student side)
        teacher_feat_dim: output dimension (teacher side)
        use_bn:           whether to include BatchNorm (default: True)
    """

    def __init__(self, student_feat_dim, teacher_feat_dim, use_bn=True):
        super().__init__()

        layers = [nn.Linear(student_feat_dim, teacher_feat_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(teacher_feat_dim))

        self.adapt = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, student_features):
        """
        Args:
            student_features: (N_active, student_feat_dim) sparse feature tensor

        Returns:
            adapted_features: (N_active, teacher_feat_dim) projected features
        """
        return self.adapt(student_features)


class MultiScaleAdaptation(nn.Module):
    """
    One adaptation layer per backbone stage for multi-scale feature distillation.

    Contains a ModuleDict of FeatureAdaptation layers, keyed by level name.

    Args:
        teacher_init_size: teacher's init_size (e.g. 32)
        student_init_size: student's init_size (e.g. 16)
        levels:            list of level names to create adaptors for
    """

    def __init__(self, teacher_init_size, student_init_size, levels=None):
        super().__init__()
        if levels is None:
            levels = MULTI_SCALE_LEVELS

        self.levels = levels
        self.adaptors = nn.ModuleDict()

        for level in levels:
            mult = LEVEL_MULTIPLIERS[level]
            s_dim = mult * student_init_size
            t_dim = mult * teacher_init_size
            self.adaptors[level] = FeatureAdaptation(s_dim, t_dim)

    def forward(self, student_features_dict):
        """
        Adapt all levels at once.

        Args:
            student_features_dict: {level_name: (N_l, student_dim)}

        Returns:
            adapted: {level_name: (N_l, teacher_dim)}
        """
        adapted = {}
        for level in self.levels:
            if level in student_features_dict:
                # IEEE 754 Half-Precision (FP16) max absolute value is 65504.0.
                # If the FP32 backbone outputs a feature like 70000.0, autocast will convert 
                # it to 'Infinity', which ruins the Linear layer's gradients. We clamp strictly.
                safe_features = torch.clamp(student_features_dict[level], min=-65504.0, max=65504.0)
                adapted[level] = self.adaptors[level](safe_features)
        return adapted


def build_multi_scale_adaptation(teacher_config, student_config, device='cuda:0'):
    """
    Build multi-scale adaptation layers (one per backbone stage).

    Args:
        teacher_config: dict with 'init_size'
        student_config: dict with 'init_size'
        device: torch device string

    Returns:
        MultiScaleAdaptation module on device
    """
    t_init = teacher_config['init_size']
    s_init = student_config['init_size']

    adaptation = MultiScaleAdaptation(t_init, s_init)
    adaptation.to(device)

    # logging
    for level in adaptation.levels:
        mult = LEVEL_MULTIPLIERS[level]
        print(f"[Adaptation] {level}: Linear({mult * s_init} -> {mult * t_init}) + BN")

    return adaptation
