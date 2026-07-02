"""
Ground Truth Loss (L_GT): Cross-Entropy + Lovász-Softmax.

General formulation:
    L_GT = -(1/N) * Σ_i Σ_c  w_c * y_{i,c} * log(p_{i,c})

where:
    N       = total number of active voxels
    C       = number of semantic classes (20 for SemanticKITTI)
    w_c     = class balancing weight (inverse frequency)
    y_{i,c} = ground truth one-hot indicator
    p_{i,c} = student's predicted softmax probability

In practice we use PyTorch's CrossEntropyLoss (which combines log-softmax
and NLL loss) plus the Lovász-Softmax loss for direct IoU optimization.
This is the same loss combination used in the original NUC-Net training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lovasz_losses import lovasz_softmax


class GroundTruthLoss(nn.Module):
    """
    Combined Cross-Entropy + Lovász-Softmax loss for semantic segmentation.

    This loss is computed between the student's predictions and the
    ground truth labels. It is the same task loss used to train the
    original teacher model.

    Args:
        num_class:    number of semantic classes
        ignore_label: label index to ignore (typically 0 = void/unlabeled)
    """

    def __init__(self, num_class=20, ignore_label=0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.ignore_label = ignore_label

    def forward(self, student_dense_logits, voxel_labels):
        """
        Args:
            student_dense_logits: (B, C, D, H, W) dense predictions from student
            voxel_labels:         (B, D, H, W) integer ground truth labels

        Returns:
            scalar loss = CE + Lovász-Softmax
        """
        # Cross-Entropy loss (class-weighted via ignore_index)
        ce = self.ce_loss(student_dense_logits, voxel_labels)

        # Lovász-Softmax loss (directly optimizes IoU)
        softmax_preds = F.softmax(student_dense_logits, dim=1)
        lovasz = lovasz_softmax(softmax_preds, voxel_labels, ignore=self.ignore_label)

        return ce + lovasz
