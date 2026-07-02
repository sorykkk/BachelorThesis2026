"""
Lambda Scheduler: controls the distillation loss weight transitions.

Training strategy from NUC-Net Distillation raw.docx:

Phase 1 (epochs 0 .. warmup_epochs):
    lambda_1 (feature)  = LARGE (e.g. 10.0) - student learns teacher's internal features
    lambda_2 (KL)       = SMALL (e.g. 1.0)
    lambda_3 (boundary) = SMALL (e.g. 0.5)

Phase 2 (epochs warmup_epochs .. total_epochs):
    lambda_1 decreases linearly -> lambda_feat_final  (e.g. 1.0)
    lambda_2 increases linearly -> lambda_kl_final    (e.g. 5.0)
    lambda_3 increases linearly -> lambda_boundary_final (e.g. 2.0)

The transition is linear for simplicity, but the scheduler is modular
so other schedules (cosine, step) can be plugged in easily.
"""


class LambdaScheduler:
    """
    Computes loss weights (lambda_1, lambda_2, lambda_3) for each training epoch.

    During warmup, weights stay at their initial values (high feature weight).
    After warmup, weights linearly interpolate to their final values.

    Args:
        lambda_feat_init:      initial lambda_1 (feature distillation)
        lambda_kl_init:        initial lambda_2 (KL divergence)
        lambda_boundary_init:  initial lambda_3 (boundary consistency)
        lambda_feat_final:     target lambda_1 after warmup
        lambda_kl_final:       target lambda_2 after warmup
        lambda_boundary_final: target lambda_3 after warmup
        warmup_epochs:         number of warmup epochs (Phase 1)
        total_epochs:          total training epochs
    """

    def __init__(self,
                 lambda_feat_init=10.0,
                 lambda_kl_init=1.0,
                 lambda_boundary_init=0.5,
                 lambda_feat_final=1.0,
                 lambda_kl_final=5.0,
                 lambda_boundary_final=2.0,
                 warmup_epochs=15,
                 total_epochs=40):
        self.lambda_feat_init = lambda_feat_init
        self.lambda_kl_init = lambda_kl_init
        self.lambda_boundary_init = lambda_boundary_init
        self.lambda_feat_final = lambda_feat_final
        self.lambda_kl_final = lambda_kl_final
        self.lambda_boundary_final = lambda_boundary_final
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # Number of epochs over which the transition happens
        self.transition_epochs = max(total_epochs - warmup_epochs, 1)

    def _interpolate(self, init_val, final_val, progress):
        """Linear interpolation: init -> final as progress goes 0 -> 1."""
        return init_val + (final_val - init_val) * progress

    def get_lambdas(self, epoch):
        """
        Get current loss weights for the given epoch.

        Args:
            epoch: current training epoch (0-indexed)

        Returns:
            dict with 'lambda_feat', 'lambda_kl', 'lambda_boundary'
        """
        if epoch < self.warmup_epochs:
            # Phase 1: warmup - use initial values (high feature weight)
            return {
                'lambda_feat': self.lambda_feat_init,
                'lambda_kl': self.lambda_kl_init,
                'lambda_boundary': self.lambda_boundary_init,
            }
        else:
            # Phase 2: transition - linearly interpolate to final values
            progress = min(
                (epoch - self.warmup_epochs) / self.transition_epochs,
                1.0  # Clamp at 1.0 after reaching total_epochs
            )
            return {
                'lambda_feat': self._interpolate(
                    self.lambda_feat_init, self.lambda_feat_final, progress),
                'lambda_kl': self._interpolate(
                    self.lambda_kl_init, self.lambda_kl_final, progress),
                'lambda_boundary': self._interpolate(
                    self.lambda_boundary_init, self.lambda_boundary_final, progress),
            }

    def __repr__(self):
        return (
            f"LambdaScheduler(\n"
            f"  Phase 1 (0..{self.warmup_epochs}): "
            f"lambda_feat={self.lambda_feat_init}, lambda_kl={self.lambda_kl_init}, "
            f"lambda_boundary={self.lambda_boundary_init}\n"
            f"  Phase 2 ({self.warmup_epochs}..{self.total_epochs}): "
            f"lambda_feat->{self.lambda_feat_final}, lambda_kl->{self.lambda_kl_final}, "
            f"lambda_boundary->{self.lambda_boundary_final}\n"
            f")"
        )
