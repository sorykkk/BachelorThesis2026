"""
Local Boundary / Consistency Loss (L_boundary).

General formulation:
    L_boundary = (1/N) * Σ_i  (1/|N(i)|) * Σ_{j ∈ N(i)}  (D_S^(i,j) - D_T^(i,j))^2

where the semantic affinity (distance) between voxel i and neighbor j is:
    D_T^(i,j) = ||sigma(Z_T^(i)/τ) - sigma(Z_T^(j)/τ)||_2^2    (teacher affinity)
    D_S^(i,j) = ||sigma(Z_S^(i)/τ) - sigma(Z_S^(j)/τ)||_2^2    (student affinity)

    N(i)   = spatial neighborhood around voxel i (adjacent active voxels)
    |N(i)| = number of valid neighbors for voxel i
    τ      = temperature scalar (same as in L_KL)

This loss forces the student to preserve the teacher's local decision
boundaries: if the teacher sees two neighboring voxels as belonging to
the same class (low affinity), the student should too, and vice versa.

Implementation note: finding neighbors in sparse 3D data is non-trivial.
We use the sparse voxel indices to build a neighbor lookup via hashing,
checking the 6 face-adjacent positions (±1 in each of x, y, z).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryConsistencyLoss(nn.Module):
    """
    Encourages the student to match the teacher's local semantic affinities
    (how similar neighboring voxels' predictions are).

    This loss captures boundary structure that pure logit matching
    (L_KL) might miss, especially at class transitions.

    Args:
        temperature: softening temperature τ (should match L_KL's τ)
    """

    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        # The 6 face-adjacent offsets in 3D (±x, ±y, ±z)
        # Sparse indices are [batch, x, y, z], so offsets are in dims 1,2,3
        self.offsets = torch.tensor([
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ], dtype=torch.long)

    def _build_neighbor_pairs(self, sparse_indices):
        """
        Find pairs of neighboring active voxels using sorted-hash lookup.

        Uses torch.searchsorted for O(N log N) lookup with O(N) memory,
        instead of a dense hash table that wastes O(max_hash) GPU memory.

        Args:
            sparse_indices: (N, 4) int tensor of [batch, x, y, z] coordinates

        Returns:
            center_idx:   (M,) indices into sparse_indices for center voxels
            neighbor_idx: (M,) indices into sparse_indices for their neighbors
        """
        device = sparse_indices.device
        N = sparse_indices.shape[0]

        if N == 0:
            return (torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        # Compute a linear hash for each active voxel.
        # Hash = batch * (X*Y*Z) + x * (Y*Z) + y * Z + z
        X_max = int(sparse_indices[:, 1].max().item()) + 2  # +2 for ±1 offset safety
        Y_max = int(sparse_indices[:, 2].max().item()) + 2
        Z_max = int(sparse_indices[:, 3].max().item()) + 2

        strides = torch.tensor(
            [X_max * Y_max * Z_max, Y_max * Z_max, Z_max, 1],
            dtype=torch.long, device=device
        )
        voxel_hashes = (sparse_indices.long() * strides).sum(dim=1)  # (N,)

        # Sort hashes for O(N)-memory searchsorted lookup
        sorted_hashes, sort_perm = torch.sort(voxel_hashes)

        # Check all 6 face-adjacent neighbors for every voxel
        offsets = self.offsets.to(device)  # (6, 4)
        center_list = []
        neighbor_list = []

        for k in range(6):
            # Compute neighbor coordinates and their hashes
            neighbor_coords = sparse_indices.long() + offsets[k].unsqueeze(0)  # (N, 4)

            # Filter out negative coordinates early
            valid_mask = (neighbor_coords >= 0).all(dim=1)
            if not valid_mask.any():
                continue

            valid_indices = torch.where(valid_mask)[0]
            neighbor_hashes = (neighbor_coords[valid_indices] * strides).sum(dim=1)

            # searchsorted: find insertion points in the sorted hash array
            positions = torch.searchsorted(sorted_hashes, neighbor_hashes)

            # Check which lookups actually found a matching hash
            in_range = positions < N
            clamped_pos = positions.clamp(max=N - 1)
            found_mask = in_range & (sorted_hashes[clamped_pos] == neighbor_hashes)

            if not found_mask.any():
                continue

            center_list.append(valid_indices[found_mask])
            neighbor_list.append(sort_perm[clamped_pos[found_mask]])

        if len(center_list) == 0:
            return (torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        center_idx = torch.cat(center_list)
        neighbor_idx = torch.cat(neighbor_list)
        return center_idx, neighbor_idx

    def forward(self, student_sparse_logits, teacher_sparse_logits, sparse_indices):
        """
        Args:
            student_sparse_logits: (N_active, C) raw logits from student
            teacher_sparse_logits: (N_active, C) raw logits from teacher
            sparse_indices:        (N_active, 4) voxel coordinates [batch, x, y, z]

        Returns:
            scalar boundary consistency loss
        """
        tau = self.temperature

        # Soften both distributions
        teacher_soft = F.softmax(teacher_sparse_logits.detach() / tau, dim=1)
        student_soft = F.softmax(student_sparse_logits / tau, dim=1)

        # Find spatially adjacent voxel pairs
        center_idx, neighbor_idx = self._build_neighbor_pairs(sparse_indices)

        if center_idx.shape[0] == 0:
            # No neighbors found; return zero loss
            return torch.tensor(0.0, device=student_sparse_logits.device, requires_grad=True)

        # Compute semantic affinities (squared L2 distance between softmax outputs)
        # Teacher affinity: D_T^(i,j) = ||σ(Z_T^i/τ) - σ(Z_T^j/τ)||²
        teacher_diff = teacher_soft[center_idx] - teacher_soft[neighbor_idx]
        D_T = (teacher_diff ** 2).sum(dim=1)    # (M,)

        # Student affinity: D_S^(i,j) = ||σ(Z_S^i/τ) - σ(Z_S^j/τ)||²
        student_diff = student_soft[center_idx] - student_soft[neighbor_idx]
        D_S = (student_diff ** 2).sum(dim=1)    # (M,)

        # Loss: MSE between student and teacher affinities
        loss = ((D_S - D_T.detach()) ** 2).mean()

        return loss
