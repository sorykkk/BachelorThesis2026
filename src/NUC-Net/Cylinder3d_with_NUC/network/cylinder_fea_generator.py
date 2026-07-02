# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None,
                 num_scales=2):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),#64
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        self.num_scales = num_scales
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression (per-scale)
        if self.fea_compre is not None:
            self.fea_compressions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.pool_dim, self.fea_compre),
                    nn.ReLU())
                for _ in range(num_scales)
            ])
            # Keep legacy attribute for backward compat with checkpoint loading
            self.fea_compression = self.fea_compressions[0]
            self.pt_fea_dim = self.fea_compre * num_scales
        else:
            self.fea_compressions = None
            self.pt_fea_dim = self.pool_dim * num_scales

    def forward(self, pt_fea, xy_ind, xy_ind_ms=None):
        cur_dev = pt_fea[0].device

        # concatenate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # process point features (shared MLP across scales)
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)

        # Scale 0: finest scale (original grid)
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        pooled_s0 = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre and self.fea_compressions is not None:
            pooled_s0 = self.fea_compressions[0](pooled_s0)

        # Multi-scale: pool at coarser scales and scatter back to finest voxels
        if self.num_scales > 1 and xy_ind_ms is not None:
            # Build coarser grid indices
            cat_pt_ind_ms = []
            for i_batch in range(len(xy_ind_ms)):
                cat_pt_ind_ms.append(F.pad(xy_ind_ms[i_batch], (1, 0), 'constant', value=i_batch))
            cat_pt_ind_coarse = torch.cat(cat_pt_ind_ms, dim=0)
            cat_pt_ind_coarse = cat_pt_ind_coarse[shuffled_ind, :]

            unq_c, unq_inv_c = torch.unique(cat_pt_ind_coarse, return_inverse=True, dim=0)[:2]
            pooled_coarse = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv_c, dim=0)[0]

            if self.fea_compre and self.fea_compressions is not None:
                pooled_coarse = self.fea_compressions[1](pooled_coarse)

            # Map coarse features back to fine voxels using a lookup:
            # For each fine voxel, find which coarse voxel its points belong to
            # We use the inverse indices: for each point, unq_inv gives fine voxel,
            # unq_inv_c gives coarse voxel. We need coarse feature per fine voxel.
            # Strategy: for each fine voxel, take the coarse feature of its first point.
            n_fine_voxels = unq.shape[0]
            fine_to_coarse = torch.zeros(n_fine_voxels, dtype=torch.long, device=cur_dev)
            # Use scatter to map: for each point, store its coarse voxel id at its fine voxel id
            # Last write wins, which is fine since all points in a fine voxel map to same coarse voxel
            fine_to_coarse.scatter_(0, unq_inv, unq_inv_c)

            pooled_s1 = pooled_coarse[fine_to_coarse]

            # Concatenate multi-scale features
            processed_pooled_data = torch.cat([pooled_s0, pooled_s1], dim=1)
        else:
            # Single-scale fallback (backward compatible)
            if self.num_scales > 1:
                # Pad with zeros if no multi-scale indices provided
                processed_pooled_data = torch.cat([pooled_s0, torch.zeros_like(pooled_s0)], dim=1)
            else:
                processed_pooled_data = pooled_s0

        return unq, processed_pooled_data
