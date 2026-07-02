"""
Combined Distillation Loss with lambda scheduling.

The total loss function:
    L_total = L_GT
            + lambda_1 * L_feat_multi_scale
            + lambda_2 * L_KL_enhanced
            + lambda_3 * L_boundary

This module combines all four loss components and applies
the lambda scheduling based on the current training epoch.
"""

import torch
import torch.nn as nn

from losses.gt_loss import GroundTruthLoss
from losses.feature_loss import MultiScaleFeatureDistillLoss
from losses.logit_loss import SoftLogitDistillLoss
from losses.boundary_loss import BoundaryConsistencyLoss
from training.lambda_scheduler import LambdaScheduler


class CombinedDistillLoss(nn.Module):
    """
    Orchestrates all distillation loss components with dynamic lambda scheduling.

    Args:
        distill_config: dict from config YAML with distillation hyperparameters
        num_class:      number of semantic classes (20 for SemanticKITTI)
        ignore_label:   label to ignore in GT loss (0 = void)
    """

    def __init__(self, distill_config, num_class=20, ignore_label=0):
        super().__init__()

        # ---- Ground truth task loss (always active) ----
        self.gt_loss = GroundTruthLoss(num_class=num_class, ignore_label=ignore_label)

        # ---- Multi-scale feature distillation (PointDistiller + PKD) ----
        self.feat_loss = MultiScaleFeatureDistillLoss(
            level_weights=distill_config.get('multi_scale_weights', [0.5, 1.0, 2.0, 4.0]),
            cosine_weight=distill_config.get('cosine_weight', 0.5),
        )

        # ---- Enhanced logit distillation (DKD + LogitKD + PointDistiller) ----
        self.logit_loss = SoftLogitDistillLoss(
            temperature=distill_config['temperature'],
            use_logit_standardization=distill_config.get('use_logit_standardization', True),
            use_entropy_weighting=distill_config.get('use_entropy_weighting', True),
            use_decoupled_kd=distill_config.get('use_decoupled_kd', True),
            dkd_alpha=distill_config.get('dkd_alpha', 1.0),
            dkd_beta=distill_config.get('dkd_beta', 8.0),
        )

        # ---- Boundary consistency loss ----
        self.boundary_loss = BoundaryConsistencyLoss(
            temperature=distill_config['temperature'],
        )

        # ---- Lambda scheduler for two-phase training ----
        self.lambda_scheduler = LambdaScheduler(
            lambda_feat_init=distill_config['lambda_feat'],
            lambda_kl_init=distill_config['lambda_kl'],
            lambda_boundary_init=distill_config['lambda_boundary'],
            lambda_feat_final=distill_config['lambda_feat_final'],
            lambda_kl_final=distill_config['lambda_kl_final'],
            lambda_boundary_final=distill_config['lambda_boundary_final'],
            warmup_epochs=distill_config['warmup_epochs'],
            total_epochs=distill_config['total_epochs'],
        )

    def forward(self, student_outputs, teacher_outputs, voxel_labels,
                adaptation_layer, epoch):
        """
        Compute the combined distillation loss.

        Args:
            student_outputs: dict from CylinderAsymDistill.forward() containing:
                - 'dense_logits':            (B, C, D, H, W) for GT loss
                - 'sparse_logits':           (N, C) for KL + boundary
                - 'sparse_indices':          (N, 4) for neighbor lookup + GT extraction
                - 'intermediate_features':   {level: (N_l, F_l)} for feature distillation
            teacher_outputs: dict from CylinderAsymDistill.forward() (same keys)
            voxel_labels:     (B, D, H, W) integer ground truth labels
            adaptation_layer: MultiScaleAdaptation module
            epoch:            current training epoch (for lambda scheduling)

        Returns:
            total_loss: scalar combined loss
            loss_dict:  dict with individual loss values (for logging)
        """
        # Get current lambda values based on training phase
        lambdas = self.lambda_scheduler.get_lambdas(epoch)
        lambda_feat = lambdas['lambda_feat']
        lambda_kl = lambdas['lambda_kl']
        lambda_boundary = lambdas['lambda_boundary']

        # ---- L_GT: Ground truth task loss (always active) ----
        l_gt = self.gt_loss(student_outputs['dense_logits'], voxel_labels)

        # ---- L_feat: Multi-scale feature distillation ----
        l_feat, feat_level_losses = self.feat_loss(
            student_outputs['intermediate_features'],
            teacher_outputs['intermediate_features'],
            adaptation_layer,
        )

        # ---- Extract per-voxel GT labels for DKD (from dense labels + sparse coords) ----
        sparse_idx = student_outputs['sparse_indices'].long()  # (N, 4) = [batch, x, y, z]
        sparse_gt = voxel_labels[
            sparse_idx[:, 0], sparse_idx[:, 1],
            sparse_idx[:, 2], sparse_idx[:, 3]
        ]  # (N,)

        # ---- L_KL: Enhanced logit distillation ----
        l_kl = self.logit_loss(
            student_outputs['sparse_logits'],
            teacher_outputs['sparse_logits'],
            gt_labels=sparse_gt,
        )

        # ---- L_boundary: Local boundary consistency ----
        l_boundary = self.boundary_loss(
            student_outputs['sparse_logits'],
            teacher_outputs['sparse_logits'],
            student_outputs['sparse_indices'],
        )

        # ---- Combine with scheduled lambdas (Ensure FP32 to avoid overflow) ----
        # Cast to float() to prevent FP16 overflow when multiplying large feature losses
        total_loss = (l_gt.float()
                      + lambda_feat * l_feat.float()
                      + lambda_kl * l_kl.float()
                      + lambda_boundary * l_boundary.float())

        # Return both the total loss and individual components for logging
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_gt': l_gt.item(),
            'loss_feat': l_feat.item(),
            'loss_kl': l_kl.item(),
            'loss_boundary': l_boundary.item(),
            'lambda_feat': lambda_feat,
            'lambda_kl': lambda_kl,
            'lambda_boundary': lambda_boundary,
        }
        # Add per-level feature losses for detailed monitoring
        loss_dict.update(feat_level_losses)

        return total_loss, loss_dict
