"""
Feature Extractor: Modified Asymm_3d_spconv backbone for distillation.

The original backbone only returns dense logits. For distillation we also need:
  - Sparse pre-logit features         (for L_feat: feature MSE distillation)
  - Sparse logits + coordinates       (for L_boundary: local consistency loss)
  - Dense logits                      (for L_GT: ground truth CE + Lovász)
  - Multi-scale intermediate features (for enhanced multi-scale distillation)

Multi-scale outputs expose features at four backbone stages:
  bottleneck   - encoder output (down4c), 16 x init_size channels
  mid_decoder  - mid-decoder (up3e)     , 8 x init_size channels
  late_decoder - late-decoder (up1e)    , 2 x init_size channels
  pre_logit    - final features (cat)   , 4 x init_size channels

References:
    PointDistiller (Luo et al., CVPR 2023): multi-scale local feature distillation
    BEVDistill (Chen et al., ICCV 2023): multi-resolution feature alignment
"""

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn

# Import building blocks from the original NUC-Net codebase
from network.segmentator_3d_asymm_spconv import (
    Asymm_3d_spconv,
    ResContextBlock,
    ResBlock,
    UpBlock,
    ReconBlock,
)


class Asymm3DSpconvDistill(Asymm_3d_spconv):
    """
    Distillation-aware variant of the Cylinder3D sparse-conv backbone.

    Inherits the full encoder-decoder architecture and only changes
    the forward() return value to expose intermediate representations
    needed by the distillation losses.

    Returns a dict:
        'dense_logits'          : (B, C, D, H, W)  - dense class predictions
        'sparse_features'       : (N, F)           - pre-logit voxel features
        'sparse_logits'         : (N, C)           - per-voxel logits (sparse)
        'sparse_indices'        : (N, 4)           - voxel coords [batch, x, y, z]
        'intermediate_features' : dict of (N_l, F_l) at four backbone stages
    """

    def forward(self, voxel_features, coors, batch_size):
        # Force backbone to FP32 to avoid spconv algorithm selection errors and NaN overflows
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure inputs are FP32
            voxel_features = voxel_features.float()
            coors = coors.int()

            # Wrap raw features into a SparseConvTensor for spconv processing
            ret = spconv.SparseConvTensor(
                voxel_features, coors, self.sparse_shape, batch_size
            )

            # ---- Encoder (progressive downsampling) ----
            ret = self.downCntx(ret)                       # context block
            down1c, down1b = self.resBlock2(ret)           # stride-2 pool (height)
            down2c, down2b = self.resBlock3(down1c)        # stride-2 pool (height)
            down3c, down3b = self.resBlock4(down2c)        # stride-2 pool (xy)
            down4c, down4b = self.resBlock5(down3c)        # stride-2 pool (xy)

            # ---- Decoder (progressive upsampling + skip connections) ----
            up4e = self.upBlock0(down4c, down4b)
            up3e = self.upBlock1(up4e, down3b)
            up2e = self.upBlock2(up3e, down2b)
            up1e = self.upBlock3(up2e, down1b)

            # ---- Refinement & feature concatenation ----
            up0e = self.ReconNet(up1e)
            # Concatenate ReconNet output with skip from last decoder stage
            # Result shape: (N_active, 4 * init_size)
            up0e = up0e.replace_feature(
                torch.cat((up0e.features, up1e.features), dim=1)
            )

            # ---- Extract pre-logit sparse features (for feature distillation) ----
            sparse_features = up0e.features           # (N_active, 4 * init_size)

            # ---- Classification head ----
            logits_sparse = self.logits(up0e)         # SubMConv3d -> (N_active, nclasses)
            sparse_logits = logits_sparse.features    # (N_active, nclasses) 
            sparse_indices = logits_sparse.indices    # (N_active, 4) = [batch, x, y, z]

            # ---- Dense logits (for GT loss computation) ----
            dense_logits = logits_sparse.dense()      # (B, nclasses, D, H, W)

            return {
                'dense_logits': dense_logits,
                'sparse_features': sparse_features,
                'sparse_logits': sparse_logits,
                'sparse_indices': sparse_indices,
                # Multi-scale intermediate features for enhanced distillation.
                # At each level the teacher and student have identical voxel
                # positions (same NUC partition + same pooling strides), so
                # features can be compared element-wise.
                'intermediate_features': {
                    'bottleneck':   down4c.features,   # (N_bottleneck, 16*S)
                    'mid_decoder':  up3e.features,     # (N_mid, 8*S)
                    'late_decoder': up1e.features,     # (N_late, 2*S)
                    'pre_logit':    sparse_features,   # (N_active, 4*S)
                },
            }


class CylinderAsymDistill(nn.Module):
    """
    Full NUC-Net model with distillation outputs.

    Combines the cylinder feature generator (MLP + NUMA multi-scale pooling)
    with the distillation-aware backbone to produce both predictions and
    intermediate features needed for distillation.
    """

    def __init__(self, cylin_model, segmentator_spconv, sparse_shape):
        super().__init__()
        self.name = "cylinder_asym_distill"
        self.cylinder_3d_generator = cylin_model         # MLP + NUMA feature gen
        self.cylinder_3d_spconv_seg = segmentator_spconv # Distillable backbone
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size,
                train_vox_ten_ms=None):
        """
        Args:
            train_pt_fea_ten: list of (N_i, 9) point feature tensors per batch item
            train_vox_ten:    list of (N_i, 3) grid index tensors (fine scale)
            batch_size:       number of items in the batch
            train_vox_ten_ms: list of (N_i, 3) grid index tensors (coarse scale, for NUMA)

        Returns:
            dict with 'dense_logits', 'sparse_features', 'sparse_logits', 'sparse_indices'
        """
        # Step 1: generate per-voxel features via MLP + NUMA multi-scale pooling
        coords, features_3d = self.cylinder_3d_generator(
            train_pt_fea_ten, train_vox_ten, xy_ind_ms=train_vox_ten_ms
        )

        # Ensure features and coords are on the same device as the spconv backbone
        # This allows hybrid execution (e.g. CPU for INT8 quantized MLP, GPU for spconv)
        spconv_device = next(self.cylinder_3d_spconv_seg.parameters()).device
        if features_3d.device != spconv_device:
            features_3d = features_3d.to(spconv_device)
            coords = coords.to(spconv_device)

        # Step 2: run features through the distillation-aware backbone
        outputs = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return outputs
