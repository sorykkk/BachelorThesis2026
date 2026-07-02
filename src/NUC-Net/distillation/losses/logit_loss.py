"""
Semantic Logit Distillation Loss (L_KL).

Three orthogonal enhancements over standard KL divergence:

**Decoupled KD**:
   Splits KL into target-class (TCKD) and non-target-class (NCKD) terms:
       L_DKD = α · TCKD + β · NCKD
   where NCKD captures inter-class similarity (the most informative signal).

**Logit Standardization**:
   Normalizes logits to zero-mean unit-variance before KL:
       Z' = (Z - μ) / sigma
   Removes batch-norm-induced bias that distorts the soft distribution.

**Entropy-based Importance Weighting**:
   Weights each voxel by the entropy of the teacher's prediction:
       w_i = H(P_T^i) / mean(H)
   Focuses the distillation signal on ambiguous / boundary voxels
   that carry the most structural information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLogitDistillLoss(nn.Module):
    """
    Enhanced KL divergence for logit-level distillation.

    Supports three optional enhancements (all enabled by default):
      - Decoupled KD (DKD): separate target / non-target class losses
      - Logit standardization: zero-mean / unit-variance normalization
      - Entropy weighting: focus on hard / boundary voxels

    Args:
        temperature:                τ for softmax softening (2.0 - 4.0)
        use_logit_standardization:  enable LogitKD normalization
        use_entropy_weighting:      enable per-voxel entropy weights
        use_decoupled_kd:           enable DKD decomposition
        dkd_alpha:                  TCKD weight (target class)
        dkd_beta:                   NCKD weight (non-target classes)
    """

    def __init__(self, temperature=3.0,
                 use_logit_standardization=True,
                 use_entropy_weighting=True,
                 use_decoupled_kd=True,
                 dkd_alpha=1.0,
                 dkd_beta=8.0):
        super().__init__()
        self.temperature = temperature
        self.use_logit_std = use_logit_standardization
        self.use_entropy = use_entropy_weighting
        self.use_dkd = use_decoupled_kd
        self.dkd_alpha = dkd_alpha
        self.dkd_beta = dkd_beta

    def _standardize_logits(self, logits):
        """Per-voxel zero-mean, unit-variance normalization (LogitKD)."""
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (logits - mean) / std

    def _compute_entropy_weights(self, teacher_probs):
        """
        Per-voxel importance weights proportional to teacher entropy.
        High-entropy voxels (ambiguous, near boundaries) get higher weight.
        """
        entropy = -(teacher_probs * teacher_probs.clamp(min=1e-8).log()).sum(dim=1)
        # Normalize so mean weight ~ 1 (preserves overall loss scale)
        weights = entropy / (entropy.mean() + 1e-8)
        return weights

    def _standard_kl(self, student_logits, teacher_logits):
        """Standard temperature-scaled KL divergence"""
        tau = self.temperature

        s = self._standardize_logits(student_logits) if self.use_logit_std else student_logits
        t = self._standardize_logits(teacher_logits.detach()) if self.use_logit_std else teacher_logits.detach()

        teacher_soft = F.softmax(t / tau, dim=1)
        student_log_soft = F.log_softmax(s / tau, dim=1)

        if self.use_entropy:
            weights = self._compute_entropy_weights(teacher_soft)
            per_voxel_kl = F.kl_div(student_log_soft, teacher_soft,
                                    reduction='none').sum(dim=1)
            loss = (per_voxel_kl * weights).mean() * (tau ** 2)
        else:
            loss = F.kl_div(student_log_soft, teacher_soft,
                            reduction='batchmean') * (tau ** 2)
        return loss

    def _decoupled_kd(self, student_logits, teacher_logits, gt_labels=None):
        """
        Decoupled Knowledge Distillation.

        Decomposes KL into:
          TCKD - binary KL on target vs. rest (captures confidence calibration)
          NCKD - KL on non-target distribution (captures inter-class structure)

        NCKD is typically more informative and gets higher weight (β > α).
        """
        tau = self.temperature

        # Use teacher argmax as pseudo-labels if GT not provided
        if gt_labels is None:
            gt_labels = teacher_logits.argmax(dim=1)

        s = self._standardize_logits(student_logits) if self.use_logit_std else student_logits
        t = self._standardize_logits(teacher_logits.detach()) if self.use_logit_std else teacher_logits.detach()

        t_probs = F.softmax(t / tau, dim=1)     # (N, C)
        s_probs = F.softmax(s / tau, dim=1)     # (N, C)

        N, C = t_probs.shape
        idx = torch.arange(N, device=t_probs.device)

        # Target class probabilities
        b_t = t_probs[idx, gt_labels].clamp(min=1e-8, max=1 - 1e-8)   # (N,)
        b_s = s_probs[idx, gt_labels].clamp(min=1e-8, max=1 - 1e-8)   # (N,)

        # TCKD: binary KL on [p_target, 1-p_target]
        tckd = (b_t * torch.log(b_t / b_s)
                + (1 - b_t) * torch.log((1 - b_t) / (1 - b_s)))       # (N,)

        # NCKD: KL on non-target class distribution (renormalized)
        mask = torch.ones_like(t_probs, dtype=torch.bool)
        mask[idx, gt_labels] = False

        t_nontarget = t_probs[mask].reshape(N, C - 1)
        s_nontarget = s_probs[mask].reshape(N, C - 1)

        # Renormalize to valid probability distributions
        t_nontarget = t_nontarget / t_nontarget.sum(dim=1, keepdim=True).clamp(min=1e-8)
        s_nontarget = s_nontarget / s_nontarget.sum(dim=1, keepdim=True).clamp(min=1e-8)

        nckd = (t_nontarget * torch.log(
            t_nontarget.clamp(min=1e-8) / s_nontarget.clamp(min=1e-8)
        )).sum(dim=1)                                                  # (N,)

        # Optional entropy weighting
        if self.use_entropy:
            weights = self._compute_entropy_weights(t_probs)
            tckd = (tckd * weights).mean()
            nckd = (nckd * weights).mean()
        else:
            tckd = tckd.mean()
            nckd = nckd.mean()

        loss = (self.dkd_alpha * tckd + self.dkd_beta * nckd) * (tau ** 2)
        return loss

    def forward(self, student_sparse_logits, teacher_sparse_logits, gt_labels=None):
        """
        Args:
            student_sparse_logits: (N_active, C) raw logits from student
            teacher_sparse_logits: (N_active, C) raw logits from teacher
            gt_labels:             (N_active,) optional per-voxel GT labels
                                   (needed for DKD; if None, uses teacher argmax)

        Returns:
            scalar KL divergence loss
        """
        if self.use_dkd:
            return self._decoupled_kd(student_sparse_logits,
                                       teacher_sparse_logits, gt_labels)
        return self._standard_kl(student_sparse_logits, teacher_sparse_logits)
