# Efficient LiDAR Panoptic Segmentation via Knowledge Distillation and Training-Free Clustering

**Bachelor Thesis — 2025**

## Abstract

Autonomous driving and robotic navigation rely on LiDAR-based 3D scene understanding to identify and distinguish every object in a point cloud. *Panoptic segmentation* unifies two sub-tasks: classifying every point into a semantic class (*stuff* like road, vegetation; *things* like cars, pedestrians), and grouping thing-class points into individual instances.

Current state-of-the-art methods achieve strong accuracy but demand large models and significant compute, limiting real-time deployment on edge hardware. This thesis investigates two complementary strategies for making LiDAR panoptic segmentation more efficient without sacrificing quality:

1. **Knowledge Distillation** - compressing a full-size semantic segmentation backbone (NUC-Net / Cylinder3D) into a student model with 50% fewer channels and ~75% fewer FLOPs, using a multi-loss distillation framework that combines multi-scale feature matching, decoupled KL divergence, and boundary affinity preservation.

2. **Training-Free Panoptic Clustering** - extending Alpine, a clustering-based panoptic head that requires no additional training, with adaptive improvements (probability-modulated edge thresholds, small cluster reassignment) to improve instance segmentation quality.

## Motivation

- **Efficiency matters**: Autonomous vehicles operate under strict latency and power budgets. Smaller models enable deployment on embedded GPUs without sacrificing safety-critical perception accuracy.
- **Training-free panoptic heads** decouple instance segmentation from backbone training, allowing any semantic backbone (including distilled ones) to be upgraded to panoptic output at zero additional training cost.
- **Knowledge distillation** is a principled compression approach: the student learns not only from ground truth labels but from the teacher's richer representations - soft probability distributions, internal feature patterns, and boundary structures.

## Contributions

| Contribution | Description |
|---|---|
| **NUC-Net Knowledge Distillation** | A multi-loss distillation pipeline (feature + logit + boundary) with two-phase scheduling, producing a 4x lighter student model |
| **AlpineAdaptive** (exploratory) | Attempted improvements to Alpine clustering (probability-modulated edges, small cluster reassignment); only the discovered k bug fix had measurable impact |
| **SOTA Benchmarking** | Evaluation and comparison of recent panoptic methods (Alpine, WaffleIron) on SemanticKITTI and nuScenes |

## Methods

### Semantic Backbones

- **WaffleIron** (Puy et al., ICCV 2023): Projects 3D points onto 2D feature planes via scatter operations, processes with standard 2D convolutions, and gathers features back - avoiding sparse 3D convolutions entirely.
- **NUC-Net** (Wang et al., TCSVT 2025): Applies non-uniform cylindrical partitioning to Cylinder3D, an asymmetric sparse convolution encoder-decoder, improving efficiency for large-scale outdoor scenes.

### Knowledge Distillation

The distillation framework trains a half-width student alongside a frozen teacher using:
- **Multi-scale feature distillation** at four backbone stages with learned adaptation layers
- **Decoupled Knowledge Distillation (DKD)** separating target-class and non-target-class KL components
- **Logit standardization** removing batch-norm-induced biases
- **Entropy-weighted importance** focusing distillation on uncertain, structurally informative voxels
- **Boundary affinity matching** preserving local decision boundaries

### Panoptic Head

- **Alpine** (Sautier et al., 2025): Constructs a kNN graph over predicted thing-class points, clusters via connected components with distance thresholds, and splits over-merged clusters using predicted bounding boxes - all without any training.
- **AlpineAdaptive** (exploratory): An attempt to improve Alpine's instance quality via probability-modulated edge thresholds and small cluster reassignment. In practice, **neither modification produced measurable gains**. The only change that affected results was discovering and fixing a variable shadowing bug where a loop variable overwrote the `k` parameter, causing kNN to use 8 neighbors instead of the intended 32. The algorithmic improvements are ineffective because Alpine's clustering already operates on high-confidence thing-class predictions where probability modulation has negligible effect, and the distance-threshold + box-splitting pipeline is robust enough that tiny noise clusters rarely survive to the final output.

## Datasets

| Dataset | Points/scan | Classes | Sequences | Benchmark |
|---|---|---|---|---|
| **SemanticKITTI** | ~120k | 19 semantic + 8 thing | 22 | Semantic + Panoptic |
| **nuScenes** | ~35k | 16 semantic | 1000 scenes | Semantic (lidarseg) |

## Repository Structure

All source code is in the [`src/`](src/) directory. See [`src/README.md`](src/README.md) for the full technical layout, build instructions, and command reference.

## Special Thanks

Special thanks to the following repositories for their invaluable open-source contributions:
- [NUC-Net](https://github.com/alanWXZ/NUC-Net)
- [Alpine](https://github.com/valeoai/Alpine/)
- [Cylinder3D](https://github.com/xinge008/Cylinder3D)
- [WaffleIron](https://github.com/valeoai/WaffleIron)
