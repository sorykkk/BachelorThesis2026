# Distillation loss components for NUC-Net
# - gt_loss:       L_GT (Cross-Entropy + Lovász-Softmax with ground truth)
# - feature_loss:  L_feat (Sparse Feature MSE distillation)
# - logit_loss:    L_KL (Temperature-scaled KL divergence)
# - boundary_loss: L_boundary (Local boundary/consistency)
# - combined_loss: Orchestrates all losses with lambda scheduling
