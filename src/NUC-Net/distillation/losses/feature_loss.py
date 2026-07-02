"""
Multi-Scale Sparse Feature Distillation Loss (L_feat).

General formulation:
    L_feat = Σ_l  α_l · [ MSE(φ_l(F_S^l), F_T^l)
                           + β · (1 - cos(φ_l(F_S^l), F_T^l)) ]

where:
    l           = backbone stage ∈ {bottleneck, mid_decoder, late_decoder, pre_logit}
    α_l         = per-level importance weight (higher for later stages)
    φ_l         = per-level learned adaptation layer
    β           = cosine similarity weight (balances MSE vs correlation)
    cos(a, b)   = cosine similarity (scale-invariant feature alignment)

Multi-scale distillation provides gradients at every backbone depth,
preventing the signal from washing out through many layers.

Cosine similarity (from PKD) makes feature alignment robust to
scale differences between teacher and student representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adaptation import MULTI_SCALE_LEVELS


class MultiScaleFeatureDistillLoss(nn.Module):
    """
    Multi-scale feature distillation loss with MSE + cosine similarity.

    Computes feature alignment at multiple backbone stages, each with its
    own adaptation layer and importance weight.

    Args:
        level_weights: list of per-level importance weights α_l
                       (indices match MULTI_SCALE_LEVELS order)
        cosine_weight: β - relative weight of cosine vs MSE component
    """

    def __init__(self, level_weights=None, cosine_weight=0.5):
        super().__init__()
        if level_weights is None:
            level_weights = [0.5, 1.0, 2.0, 4.0]
        self.level_weights = level_weights
        self.cosine_weight = cosine_weight
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, student_intermediate, teacher_intermediate, adaptation):
        """
        Args:
            student_intermediate: dict {level_name: (N_l, student_dim)}
            teacher_intermediate: dict {level_name: (N_l, teacher_dim)}
            adaptation:           MultiScaleAdaptation module

        Returns:
            total_loss: scalar multi-scale feature loss
            level_losses: dict {f'feat_{level}': float} for logging
        """
        level_losses = {}
        
        # Calculate loss in FP32 to prevent overflow (important if MSE is high)
        with torch.cuda.amp.autocast(enabled=False):
            # Run adaptation in FP32 to prevent FP16 overflow (features can be large)
            student_fp32 = {k: v.float() for k, v in student_intermediate.items()}
            adapted = adaptation(student_fp32)

            total_loss = torch.tensor(0.0, device=next(iter(adapted.values())).device)

            for i, level in enumerate(MULTI_SCALE_LEVELS):
                if level not in adapted:
                    continue

                weight = self.level_weights[i] if i < len(self.level_weights) else 1.0
                s_feat = adapted[level].float()
                t_feat = teacher_intermediate[level].detach().float()

                # MSE component: ||phi(F_S) - F_T||^2
                mse_loss = self.mse(s_feat, t_feat)

                # Cosine similarity component: 1 - cos(phi(F_S), F_T)
                # Scale-invariant alignment (PKD)
                if self.cosine_weight > 0 and s_feat.shape[0] > 0:
                    cos_sim = F.cosine_similarity(s_feat, t_feat, dim=1)
                    cos_loss = (1.0 - cos_sim).mean()
                else:
                    cos_loss = torch.tensor(0.0, device=s_feat.device)

                level_loss = mse_loss + self.cosine_weight * cos_loss
                total_loss = total_loss + weight * level_loss
                level_losses[f'feat_{level}'] = level_loss.item()

        return total_loss, level_losses
