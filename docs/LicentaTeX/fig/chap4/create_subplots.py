import matplotlib.pyplot as plt
from PIL import Image

def plot_images(images, orientation="horizontal", save_path=None, dpi=300):
    n = len(images)

    if n == 2:
        if orientation == "horizontal":
            rows, cols = 1, 2
            figsize = (12, 6)
        elif orientation == "vertical":
            rows, cols = 2, 1
            figsize = (6, 12)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

    elif n == 4:
        rows, cols = 2, 2
        # FIX 1: Change to a rectangular figsize to better match the images' aspect ratio
        figsize = (12, 8) 

    else:
        raise ValueError("Only 2 or 4 images are supported.")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    axes = axes.flatten() if n > 1 else [axes]

    for ax, (img_path, caption) in zip(axes, images):
        img = Image.open(img_path)

        ax.imshow(img)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel(caption, fontsize=14)

    # FIX 2: Explicitly set h_pad (vertical spacing) to a small number
    plt.tight_layout(h_pad=0.5, w_pad=1.0)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

plot1 = [
    ("11_loss_gt.png", "(a)"),
    ("12_loss_feat.png", "(b)"),
    ("13_loss_kl.png", "(c)"),
    ("14_loss_boundary.png", "(d)")
]

plot2 = [
    ("12_loss_feat.png", "(a)"),
    ("22_loss_feat_final.png", "(b)"),
]

plot3 = [
    ("44_feat_bottleneck.png", "(a)"),
    ("42_feat_mid_decoder.png", "(b)"),
    ("41_feat_late_decoder.png", "(c)"),
    ("43_feat_pre_logit.png", "(d)"),
]

plot4 = [
    ("44_feat_bottleneck.png", "(a)"),
    ("32_feat_bottleneck_final.png", "(b)")
]

plot5 = [
    ("51_loss_total.png", "(a)"),
    ("52_loss_total_final.png", "(b)")
]

plot_images(plot1, save_path="./1_all_losses.png")
plot_images(plot2, orientation="horizontal", save_path="./2_loss_feat.png")
plot_images(plot3, save_path="./3_all_feat.png")
plot_images(plot4, orientation="horizontal", save_path="./4_feat_bottleneck.png")
plot_images(plot5, orientation="horizontal", save_path="./5_loss_total.png")