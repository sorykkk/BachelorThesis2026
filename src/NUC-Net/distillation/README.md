# NUC-Net Knowledge Distillation

Distills a full-size NUC-Net teacher model into a **50% smaller** student model while minimizing the mIoU gap on SemanticKITTI, using state-of-the-art techniques from recent knowledge distillation literature.

## Architecture

Both teacher and student share the same overall structure:
- **NUC** (Non-Uniform Cylindrical partitioning) - same partition parameters
- **NUMA** (Non-Uniform Multi-scale Aggregation) - 2-scale feature pooling
- **Cylinder3D** sparse convolution backbone - encoder-decoder with skip connections

The student's backbone channels are halved (width reduction, not depth):

| Component | Teacher (`init_size=32`) | Student (`init_size=16`) |
|---|---|---|
| Feature compression | 32 per scale (64 total) | 16 per scale (32 total) |
| Encoder channels | 32 -> 64 -> 128 -> 256 -> 512 | 16 -> 32 -> 64 -> 128 -> 256 |
| Decoder channels | 512 -> 256 -> 128 -> 64 | 256 -> 128 -> 64 -> 32 |
| Pre-logit features | 128 | 64 |
| Est. FLOPs reduction | - | ~75% |
| Est. params reduction | 55.89 M | ~75% |

## Loss Function and Considerations

$$\mathcal{L}_{total} = \mathcal{L}_{GT} + \lambda_1 \mathcal{L}_{feat}^{ms} + \lambda_2 \mathcal{L}_{KL}^{enh} + \lambda_3 \mathcal{L}_{boundary}$$

| Loss | Description |
|---|---|
| $\mathcal{L}_{GT}$ | CE + Lovász-Softmax |
| $\mathcal{L}_{feat}^{ms}$ | Multi-scale feature distillation with MSE + cosine similarity at 4 backbone stages |
| $\mathcal{L}_{KL}^{enh}$ | Enhanced logit distillation with DKD + logit standardization + entropy weighting |
| $\mathcal{L}_{boundary}$ | Local boundary affinity matching with searchsorted-based O(N) memory neighbor lookup |

### Multi-Scale Feature Distillation

Instead of distilling only the final pre-logit features, we extract and match features at **four backbone stages**:

$$\mathcal{L}_{feat}^{ms} = \sum_{l=1}^{4} \alpha_l \left[ \text{MSE}(\phi_l(F_S^l), F_T^l) + \beta \cdot (1 - \cos(\phi_l(F_S^l), F_T^l)) \right]$$

| Stage | Location | Teacher dim | Student dim | α weight |
|---|---|---|---|---|
| Bottleneck | `down4c` (encoder output) | 512 | 256 | 0.5 |
| Mid-decoder | `up3e` (after upBlock1) | 256 | 128 | 1.0 |
| Late-decoder | `up1e` (after upBlock3) | 64 | 32 | 2.0 |
| Pre-logit | `cat(up0e, up1e)` | 128 | 64 | 4.0 |

Each stage has its own learned adaptation layer `φ_l: Linear + BN` (no ReLU - teacher features from LeakyReLU can be negative).

**Cosine similarity** (β=0.5) provides scale-invariant feature alignment on top of MSE, following PKD [3].

**Why it helps:** Multi-scale distillation provides gradient signal at every backbone depth. Early layers capture geometric patterns (edges, surfaces), middle layers capture local semantic groupings, and late layers capture high-level class information. Distilling only the last layer provides a weak signal through the full encoder-decoder chain.

### Decoupled Knowledge Distillation (DKD)

Standard KL divergence treats all logit positions equally. DKD [2] decomposes KL into two components:

$$\mathcal{L}_{DKD} = \alpha \cdot \text{TCKD} + \beta \cdot \text{NCKD}$$

- **TCKD** (Target Class KD): binary KL on target-vs-rest probability - captures confidence calibration
- **NCKD** (Non-Target Class KD): KL on the renormalized non-target distribution - captures inter-class similarity structure

NCKD is the most informative signal and gets higher weight (β=8.0 vs α=1.0).

### Logit Standardization (LogitKD)

Before computing KL divergence, logits are normalized to zero-mean, unit-variance [5]:

$$Z' = \frac{Z - \mu}{\sigma}$$

This removes batch-normalization-induced biases that artificially distort the soft probability distribution, leading to a purer distillation signal.

### Entropy-Based Importance Weighting

Each voxel is weighted by the entropy of the teacher's prediction [1]:

$$w_i = \frac{H(P_T^{(i)})}{\overline{H}}$$

High-entropy voxels (near class boundaries, rare classes, ambiguous regions) get more distillation signal. Low-entropy voxels (confident interior regions) contribute less. This focuses learning on the most structurally informative parts of the scene.

### No-ReLU Adaptation Layers

Since teacher features pass through LeakyReLU and can be negative, the adaptation is `Linear + BN` only, without using any `ReLU`.

### Training Schedule

- **Phase 1** (epochs 0–15): Large `λ₁` (feature learning), small `λ₂`, `λ₃`
- **Phase 2** (epochs 15–40): Decrease `λ₁`, increase `λ₂` and `λ₃` (logit/boundary alignment)

## Usage

### Prerequisites

Same environment as the main NUC-Net codebase (spconv, torch_scatter, etc.).
A pretrained teacher checkpoint must exist at the path specified in the config.

### Training

```bash
cd src/NUC-Net/distillation

# Start distillation training
python training/train_distill.py -y config/distill_semantickitti.yaml

# Resume from checkpoint
python training/train_distill.py -y config/distill_semantickitti.yaml --resume
```

### Evaluation

```bash
# Evaluate student model
python evaluation/eval_student.py -y config/distill_semantickitti.yaml

# Compare student vs teacher
python evaluation/eval_student.py -y config/distill_semantickitti.yaml --compare_teacher

# Override student checkpoint
python evaluation/eval_student.py -y config/distill_semantickitti.yaml -m ./checkpoints/student_best.pt
```

## Theory

Knowledge distillation transfers knowledge from a large, accurate **teacher** model to a smaller, faster **student** model. The student learns not just from ground truth labels, but from the teacher's soft predictions and internal representations:

1. **Feature Distillation** (`L_feat`): Forces the student to match the teacher's internal feature representations at each active voxel. An adaptation layer `φ` bridges the dimension gap.

2. **Soft Label Distillation** (`L_KL`): Uses temperature-softened distributions to transfer inter-class relationships (e.g., "car is more similar to truck than to tree").

3. **Boundary Consistency** (`L_boundary`): Preserves the teacher's decision boundaries by matching local semantic affinities between neighboring voxels.

## References

1. **PointDistiller** - Luo et al., *"PointDistiller: Structured Knowledge Distillation Towards Efficient and Compact 3D Detection"*, CVPR 2023.
   Multi-scale local feature distillation and difficulty-based reweighting for 3D sparse models.

2. **DKD** - Zhao et al., *"Decoupled Knowledge Distillation"*, CVPR 2022.
   Decomposes KL into target-class and non-target-class components; NCKD is the primary signal.

3. **PKD** - Cao et al., *"PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient"*, ECCV 2022.
   Pearson correlation / cosine similarity for scale-invariant feature alignment.

4. **BEVDistill** - Chen et al., *"BEVDistill: Cross-Modal BEV Distillation for Multi-View 3D Object Detection"*, ICCV 2023.
   Multi-resolution feature distillation across backbone stages.

5. **LogitKD** - Sun et al., *"Logit Standardization in Knowledge Distillation"*, AAAI 2024.
   Zero-mean unit-variance logit normalization removes BN-induced bias.

6. **Hinton et al.** - *"Distilling the Knowledge in a Neural Network"*, NeurIPS Workshop 2015.
   Original knowledge distillation framework with temperature-scaled soft labels.
