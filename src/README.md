# LiDAR Panoptic Segmentation - Source Code

This directory contains all source code, experiments, data links, and tooling for the thesis on efficient LiDAR panoptic segmentation.

## Directory Structure

Essential directory structure:
```
src/
├── data/                    # Dataset symlinks / directories
│   ├── semantickitti/       # SemanticKITTI (22 sequences, ~43k scans)
│
├── experiments/             # Modified copies used for training & evaluation
│   ├── Alpine/              # Alpine panoptic head (vanilla)
│   ├── AlpineAdaptive/      # Alpine with probability-modulated edges,
│   │                        #   small cluster reassignment, and k bug fix
│   ├── NUC-Net/             # NUC-Net training (baseline training)
│   ├── WaffleIron/          # WaffleIron with training configs & saved models
│   └── README.md            # Commands & instructions for all experiments
│
├── NUC-Net/                 # Primary NUC-Net codebase (teacher + distillation)
│   ├── Cylinder3d_with_NUC/ # Full teacher backbone (init_size=32)
│   │   ├── builder/         # Model & loss builders
│   │   ├── config/          # YAML configs per dataset
│   │   ├── dataloader/      # SemanticKITTI / nuScenes data loading with NUC
│   │   ├── network/         # Cylinder3D + NUMA modules
│   │   └── train_cyl_sem.py # Main training entry point
│   └── distillation/        # Knowledge distillation pipeline
│       ├── config/          # distill_semantickitti.yaml
│       ├── models/          # Student/teacher wrappers, feature extractors
│       ├── losses/          # Multi-scale feature, DKD, boundary losses
│       ├── training/        # train_distill.py (main entry)
│       ├── evaluation/      # eval_student.py (evaluate student's performance)
│       └── README.md        # Full distillation documentation
│
├── scripts/
│   └── profiling/           # Analytical FLOPs estimator for sparse conv models
│       ├── configs/         # Model configs for profiling
│       └── flops_estimate.py
│
└── code-demo/               # Demo code for the usage of the distilled model
```

## Semantic Segmentation Backbones

| Backbone | Architecture | Key idea |
|---|---|---|
| **WaffleIron** | 3D -> 2D scatter, 2D conv, 2D -> 3D gather | Projection-based; avoids sparse 3D conv |
| **NUC-Net (Cylinder3D)** | Sparse 3D conv encoder-decoder | Non-uniform cylindrical partitioning for efficiency |
| **NUC-Net Student** | Same architecture, 50% channel width | Knowledge distillation from full teacher |

## Panoptic Segmentation Head

| Method | Approach |
|---|---|
| **Alpine** | Training-free kNN graph clustering + box splitting |
| **AlpineAdaptive** | Alpine + attempted improvements (see below); only the k bug fix had effect |

AlpineAdaptive is backward-compatible with vanilla Alpine (`probs=None`, `min_cluster_size=0`). The probability-modulated edges and small cluster reassignment were exploratory and did not produce measurable quality gains - see the [AlpineAdaptive section](#alpineadaptive-exploratory) for analysis.

## Knowledge Distillation (NUC-Net)

The student (50% width, ~75% FLOPs reduction) is trained with a composite loss:

$$\mathcal{L} = \mathcal{L}_{CE+Lov} + \lambda_1 \mathcal{L}_{feat}^{ms} + \lambda_2 \mathcal{L}_{KL}^{DKD} + \lambda_3 \mathcal{L}_{boundary}$$

- **Multi-scale feature distillation** at 4 backbone stages (bottleneck -> pre-logit)
- **Decoupled KD** with logit standardization and entropy weighting
- **Boundary affinity matching** via searchsorted-based O(N) neighbor lookup
- **Two-phase schedule**: feature-focus warmup -> logit/boundary ramp

See [NUC-Net/distillation/README.md](NUC-Net/distillation/README.md) for full details.

## AlpineAdaptive (Exploratory)

Three modifications were attempted on the Alpine clustering head. Only the bug fix had a measurable effect:

1. **Probability-modulated edge thresholding**: $th_{eff}(i,j) = th \cdot \sqrt{p_i \cdot p_j}$ - scales edge thresholds by prediction confidence. **No measurable impact**: Alpine already filters to high-confidence thing-class points before clustering, so the probability modulation factor $\sqrt{p_i \cdot p_j}$ stays close to 1.0 for nearly all edges and does not meaningfully alter connectivity.
2. **Small cluster reassignment**: Clusters with < `min_cluster_size` points are merged into their nearest large cluster via kNN. **No measurable impact**: Alpine's distance-threshold filtering and box-based split step already suppress most noise fragments; the few remaining tiny clusters are too small to affect PQ/RQ metrics.
3. **k parameter bug fix** (**only effective change**): A loop variable `k` in `__init__` was shadowing the `k` parameter, causing `self.k` to be set to the last dict key (e.g., 8 on SemanticKITTI) instead of the intended 32. Fixing this restored proper kNN graph connectivity and was the sole source of quality improvement.

## Profiling

`scripts/profiling/flops_estimate.py` provides analytical FLOPs/parameter estimation for sparse conv architectures (Cylinder3D, WaffleIron) where standard profilers (torchprofile, fvcore) fail due to `spconv` and scatter/gather operations.

## Quick Start

All training and evaluation commands are documented in [experiments/README.md](experiments/README.md).

### Environment

You can easily set up the `nucnet-env` Conda environment across different machines using the provided `nucnet_environment.yml` file located in the root of this project:
```bash
conda env create -f nucnet_environment.yml
conda activate nucnet-env
```
This environment can be also used for Alpine as it uses the same dependencies.

Common core dependencies:
- PyTorch >= 1.10
- spconv-cu11x (for NUC-Net / Cylinder3D)
- torch-scatter

### Datasets

SemanticKITTI should be placed (or symlinked) under `data/`:
```
data/semantickitti/dataset/sequences/{00..21}/
```
