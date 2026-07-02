# FLOPs Estimation

Analytical FLOPs profiler for **Cylinder3D** (NUC-Net) and **WaffleIron** on SemanticKITTI.

## Why analytical?

Standard profilers (fvcore, thop) cannot trace:
- **Cylinder3D** - uses `spconv` sparse convolutions (`SubMConv3d`, `SparseConv3d`)
- **WaffleIron** - uses scatter/gather-based 3D -> 2D projections

This script computes FLOPs by walking each layer analytically. Sparse conv FLOPs depend on the number of active voxels, which is data-dependent - defaults represent typical SemanticKITTI scans.

## Configs

Each model variant has a YAML config in `configs/`:

| Config | Model | Dataset |
|---|---|---|
| `cylinder3d_singlescan.yaml` | Cylinder3D Teacher (init_size=32, 2 scales) | SemanticKITTI SS |
| `cylinder3d_distilled.yaml` | Cylinder3D Distilled Student (init_size=16, 2 scales) | SemanticKITTI SS |
| `cylinder3d_multiscan.yaml` | Cylinder3D (init_size=16, 1 scale) | SemanticKITTI MS |
| `waffleiron_kitti.yaml` | WaffleIron-48-256 | SemanticKITTI |

Config structure:
```yaml
arch: cylinder3d          # or waffleiron
model:
  init_size: 32           # architecture parameters
  ...
profiling:
  num_points: 120000      # default point count
  num_voxels: 40000       # default active voxels (Cylinder3D only)
```

## Usage

```bash
cd scripts/profiling

# Run with a config
python flops_estimate.py --config configs/cylinder3d_singlescan.yaml
python flops_estimate.py --config configs/cylinder3d_distilled.yaml
python flops_estimate.py --config configs/cylinder3d_multiscan.yaml
python flops_estimate.py --config configs/waffleiron_kitti.yaml

# Override point/voxel counts
python flops_estimate.py --config configs/cylinder3d_singlescan.yaml --num-points 150000 --num-voxels 50000

# Verify parameter count against a checkpoint
python flops_estimate.py --config configs/cylinder3d_singlescan.yaml --checkpoint path/to/ckpt.pt
```

## CLI Arguments

| Argument | Description |
|---|---|
| `--config` | Path to a profiling YAML config (required) |
| `--num-points` | Override the number of input points |
| `--num-voxels` | Override active voxels at finest level (Cylinder3D only) |
| `--checkpoint` | Path to `.pt`/`.pth` file - counts real parameters for comparison |

## Output

The script prints:
- Input assumptions (points, voxels, grid shape, channels)
- Per-layer FLOPs breakdown
- Subtotals by component (feature generator / backbone / embedding / classifier)
- Total FLOPs
- Estimated parameter count (and comparison with checkpoint if provided)

## FLOPs Convention

FLOPs = multiply-accumulate x 2 (each MAC = 1 multiply + 1 add).
