#!/usr/bin/env python3
"""
FLOPs Estimation for NUC-Net Cylinder3D (single-scan / multi-scan) and WaffleIron.

Computes analytical FLOPs estimates because:
  - Cylinder3D uses spconv (SubMConv3d / SparseConv3d) which standard profilers
    like fvcore or thop cannot trace.
  - WaffleIron uses scatter/gather projections that are also not traceable.

FLOPs are counted as multiply-accumulate operations x2 (standard convention).
For sparse convolutions the count depends on the number of active voxels,
which is data-dependent.  Default values represent typical SemanticKITTI scans.

Each model has a dedicated profiling config under scripts/configs/.
You can also override point/voxel counts from the CLI.

Usage examples:
    python flops_estimate.py --config configs/cylinder3d_singlescan.yaml
    python flops_estimate.py --config configs/cylinder3d_multiscan.yaml --num-points 500000
    python flops_estimate.py --config configs/waffleiron_kitti.yaml
    python flops_estimate.py --config configs/cylinder3d_singlescan.yaml --checkpoint path/to/ckpt.pt
"""

import argparse
import os
import yaml


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(n, unit=""):
    """Format a large number with SI prefix."""
    for threshold, suffix in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if n >= threshold:
            return f"{n / threshold:.2f} {suffix}{unit}"
    return f"{n:.0f} {unit}"


def section(title, width=70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Sparse-conv layer FLOPs (Cylinder3D building blocks)
# ---------------------------------------------------------------------------

def _subm_flops(n_active, c_in, c_out, kernel_size):
    """SubMConv3d: output keeps the same set of active voxels."""
    if isinstance(kernel_size, (list, tuple)):
        kv = 1
        for k in kernel_size:
            kv *= k
    else:
        kv = kernel_size ** 3
    return n_active * c_in * c_out * kv * 2


def _strided_flops(n_out, c_in, c_out, kernel_size=3):
    """SparseConv3d / SparseInverseConv3d: count against output voxels."""
    kv = kernel_size ** 3
    return n_out * c_in * c_out * kv * 2


def _res_context_flops(n, ci, co):
    """ResContextBlock(ci, co): two parallel paths merged by addition."""
    # Path A: conv1x3(ci,co) -> conv3x1(co,co)
    # Path B: conv3x1(ci,co) -> conv1x3(co,co)
    return (
        _subm_flops(n, ci, co, (1, 3, 3))
        + _subm_flops(n, co, co, (3, 1, 3))
        + _subm_flops(n, ci, co, (3, 1, 3))
        + _subm_flops(n, co, co, (1, 3, 3))
    )


def _res_block_flops(n_in, n_out, ci, co, pooling=True):
    """ResBlock(ci, co) with optional strided SparseConv3d pool."""
    f = (
        _subm_flops(n_in, ci, co, (3, 1, 3))
        + _subm_flops(n_in, co, co, (1, 3, 3))
        + _subm_flops(n_in, ci, co, (1, 3, 3))
        + _subm_flops(n_in, co, co, (3, 1, 3))
    )
    if pooling:
        f += _strided_flops(n_out, co, co, 3)
    return f


def _up_block_flops(n_low, n_high, ci, co):
    """UpBlock(ci, co): conv at low res, inverse-conv to high res, 3 convs."""
    return (
        _subm_flops(n_low, ci, co, (3, 3, 3))       # trans_dilao @ low
        + _strided_flops(n_high, co, co, 3)           # inverse conv -> high
        + _subm_flops(n_high, co, co, (1, 3, 3))     # conv1 @ high
        + _subm_flops(n_high, co, co, (3, 1, 3))     # conv2 @ high
        + _subm_flops(n_high, co, co, (3, 3, 3))     # conv3 @ high
    )


def _recon_block_flops(n, ci, co):
    """ReconBlock(ci, co): three 1D-kernel sparse convs + element-wise mul."""
    return (
        _subm_flops(n, ci, co, (3, 1, 1))
        + _subm_flops(n, ci, co, (1, 3, 1))
        + _subm_flops(n, ci, co, (1, 1, 3))
        + n * ci  # element-wise multiply (negligible but included)
    )


# ---------------------------------------------------------------------------
# Cylinder3D  -  full model FLOPs
# ---------------------------------------------------------------------------

def cylinder3d_flops(cfg, num_points, num_voxels):
    s = cfg["init_size"]
    ns = cfg.get("num_scales", 1)
    ci_spconv = cfg["num_input_features"] * ns
    nc = cfg["num_class"]
    fea_dim = cfg["fea_dim"]
    out_fea = cfg["out_fea_dim"]
    fea_compre = cfg["num_input_features"]

    detail = {}

    # ---- Feature generator (cylinder_fea) ----
    mlp = [(fea_dim, 64), (64, 128), (128, 256), (256, out_fea)]
    detail["PPmodel MLP"] = sum(num_points * ci * co * 2 for ci, co in mlp)
    detail["fea_compression"] = ns * num_voxels * out_fea * fea_compre * 2
    fea_total = detail["PPmodel MLP"] + detail["fea_compression"]

    # ---- Estimate active-voxel counts per encoder level ----
    # Empirical reduction factors for sparse strided conv on SemanticKITTI:
    #   height_pooling  (stride 2,2,2) -> ~20 % voxels survive
    #   no height_pool  (stride 2,2,1) -> ~30 % voxels survive
    hp = [True, True, False, False]  # hardcoded in Asymm_3d_spconv
    n = [num_voxels]
    for h in hp:
        n.append(int(n[-1] * (0.20 if h else 0.30)))

    # ---- 3-D backbone (Asymm_3d_spconv) ----
    detail["downCntx"]   = _res_context_flops(n[0], ci_spconv, s)
    detail["resBlock2"]  = _res_block_flops(n[0], n[1], s, 2 * s)
    detail["resBlock3"]  = _res_block_flops(n[1], n[2], 2 * s, 4 * s)
    detail["resBlock4"]  = _res_block_flops(n[2], n[3], 4 * s, 8 * s)
    detail["resBlock5"]  = _res_block_flops(n[3], n[4], 8 * s, 16 * s)
    detail["upBlock0"]   = _up_block_flops(n[4], n[3], 16 * s, 16 * s)
    detail["upBlock1"]   = _up_block_flops(n[3], n[2], 16 * s, 8 * s)
    detail["upBlock2"]   = _up_block_flops(n[2], n[1], 8 * s, 4 * s)
    detail["upBlock3"]   = _up_block_flops(n[1], n[0], 4 * s, 2 * s)
    detail["ReconNet"]   = _recon_block_flops(n[0], 2 * s, 2 * s)
    detail["logits"]     = _subm_flops(n[0], 4 * s, nc, 3)  # cat(recon, up1) -> 4*s

    backbone_total = sum(v for k, v in detail.items()
                         if k not in ("PPmodel MLP", "fea_compression"))
    total = fea_total + backbone_total
    return detail, fea_total, backbone_total, total, n


# ---------------------------------------------------------------------------
# WaffleIron  -  full model FLOPs
# ---------------------------------------------------------------------------

def waffleiron_flops(cfg, N):
    Ci = cfg["input_channels"]       # 5
    C  = cfg["feat_channels"]        # 256 / 384
    D  = cfg["depth"]                # 48
    nc = cfg["nb_class"]
    grids = cfg["grids_size"]
    K  = cfg.get("neighbors", 16)

    detail = {}

    # ---- Embedding ----
    detail["embed conv1"]  = N * Ci * C * 2                   # Conv1d(Ci,C,1)
    detail["embed conv2a"] = K * N * Ci * C * 2               # Conv2d(Ci,C,1)
    detail["embed conv2b"] = K * N * C * C * 2                # Conv2d(C,C,1)
    detail["embed final"]  = N * 2 * C * C * 2                # Conv1d(2C,C,1)
    emb_total = sum(detail[k] for k in detail)

    # ---- Backbone (D layers) ----
    cm_total = 0
    sm_total = 0
    for d in range(D):
        # ChannelMix: BN + Conv1d(C,C,1) + ReLU + Conv1d(C,C,1) + LayerScale
        cm = N * C * C * 2 + N * C * C * 2 + N * C * 2
        cm_total += cm

        # SpatialMix: BN + project + depthwise Conv2d×2 + LayerScale + inflate
        H, W = grids[d % len(grids)]
        sm = H * W * C * 9 * 2 + H * W * C * 9 * 2 + N * C * 2
        sm_total += sm

    detail["ChannelMix (x {})".format(D)] = cm_total
    detail["SpatialMix (x {})".format(D)] = sm_total
    backbone_total = cm_total + sm_total

    # ---- Classification head ----
    detail["classif"] = N * C * nc * 2
    total = emb_total + backbone_total + detail["classif"]
    return detail, emb_total, backbone_total, total


# ---------------------------------------------------------------------------
# Analytical parameter counts (for verification against checkpoints)
# ---------------------------------------------------------------------------

def cylinder3d_params(cfg):
    s = cfg["init_size"]
    ns = cfg.get("num_scales", 1)
    ci_sp = cfg["num_input_features"] * ns
    nc = cfg["num_class"]
    fd = cfg["fea_dim"]
    ofd = cfg["out_fea_dim"]
    fc = cfg["num_input_features"]

    p = 0
    # PPmodel MLP
    for ci, co in [(fd, 64), (64, 128), (128, 256), (256, ofd)]:
        p += ci * co + co + co * 2          # Linear + BN
    p += fd * 2                              # leading BN

    # Feature compression (per scale)
    p += ns * (ofd * fc + fc)                # Linear(ofd, fc) + bias (inside ReLU seq)

    # Sparse-conv helpers
    def sp(ci, co, kv): return ci * co * kv
    def bn(c): return c * 2

    def rctx(ci, co):
        return (sp(ci, co, 9) + bn(co)) * 2 + (sp(co, co, 9) + bn(co)) * 2

    def rblk(ci, co, pool=True):
        t = (sp(ci, co, 9) + bn(co)) * 2 + (sp(co, co, 9) + bn(co)) * 2
        if pool:
            t += sp(co, co, 27)
        return t

    def ublk(ci, co):
        return (sp(ci, co, 27) + bn(co)          # trans_dilao
                + sp(co, co, 27)                  # up_subm
                + sp(co, co, 9) + bn(co)          # conv1
                + sp(co, co, 9) + bn(co)          # conv2
                + sp(co, co, 27) + bn(co))        # conv3

    def rcn(ci, co):
        return (sp(ci, co, 3) + bn(co)) * 3

    p += rctx(ci_sp, s)
    p += rblk(s, 2*s)
    p += rblk(2*s, 4*s)
    p += rblk(4*s, 8*s)
    p += rblk(8*s, 16*s)
    p += ublk(16*s, 16*s)
    p += ublk(16*s, 8*s)
    p += ublk(8*s, 4*s)
    p += ublk(4*s, 2*s)
    p += rcn(2*s, 2*s)
    p += sp(4*s, nc, 27) + nc  # logits (bias=True)
    return p


def waffleiron_params(cfg):
    Ci = cfg["input_channels"]
    C  = cfg["feat_channels"]
    D  = cfg["depth"]
    nc = cfg["nb_class"]

    p = 0
    # Embedding
    p += Ci * 2                          # norm BN1d
    p += Ci * C + C                      # conv1
    p += Ci * 2 + Ci * C + C * 2 + C * C  # conv2 (BN+Conv+BN+Conv, no bias)
    p += 2 * C * C + C                   # final

    # Backbone
    for _ in range(D):
        # ChannelMix
        p += C * 2 + (C * C + C) + (C * C + C) + C
        # SpatialMix
        p += C * 2 + (C * 9 + C) + (C * 9 + C) + C

    # Classification
    p += C * nc + nc
    return p


def params_from_checkpoint(path):
    """Load checkpoint and count total parameters."""
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # older PyTorch versions don't support weights_only
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "net", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    if isinstance(ckpt, dict):
        return sum(t.numel() for t in ckpt.values())
    # full model object
    return sum(p.numel() for p in ckpt.parameters())


# ---------------------------------------------------------------------------
# YAML config loaders
# ---------------------------------------------------------------------------

# Default assumptions for values not present in YAML files
CYLINDER3D_DEFAULTS = {
    "num_scales": 1,
    "default_num_points": 120_000,
    "default_num_voxels": 40_000,
}

WAFFLEIRON_DEFAULTS = {
    "neighbors": 16,
    "default_num_points": 20_000,
}


def detect_arch(raw):
    """Auto-detect architecture type from the YAML structure."""
    # Profiling config format: explicit 'arch' key
    if "arch" in raw:
        return raw["arch"]
    # Original project YAML formats
    if "model_params" in raw:
        return "cylinder3d"
    if "waffleiron" in raw:
        return "waffleiron"
    return None


def load_cylinder3d_config(path):
    """Read a Cylinder3D YAML and return a flat config dict."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # Support both profiling format (model:) and project format (model_params:)
    mp = raw.get("model") or raw["model_params"]
    prof = raw.get("profiling", {})
    cfg = {
        "label": f"Cylinder3D  ({os.path.basename(path)})",
        "init_size": mp["init_size"],
        "num_input_features": mp["num_input_features"],
        "num_scales": mp.get("num_scales", CYLINDER3D_DEFAULTS["num_scales"]),
        "num_class": mp["num_class"],
        "fea_dim": mp["fea_dim"],
        "out_fea_dim": mp["out_fea_dim"],
        "output_shape": mp["output_shape"],
        "default_num_points": prof.get("num_points", CYLINDER3D_DEFAULTS["default_num_points"]),
        "default_num_voxels": prof.get("num_voxels", CYLINDER3D_DEFAULTS["default_num_voxels"]),
    }
    return cfg


def load_waffleiron_config(path):
    """Read a WaffleIron YAML and return a flat config dict."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # Support both profiling format (model:) and project format (waffleiron: + embedding: + classif:)
    if "model" in raw:
        mp = raw["model"]
        prof = raw.get("profiling", {})
        cfg = {
            "label": f"WaffleIron  ({os.path.basename(path)})",
            "input_channels": mp["input_channels"],
            "feat_channels": mp["feat_channels"],
            "depth": mp["depth"],
            "nb_class": mp["nb_class"],
            "grids_size": mp["grids_size"],
            "neighbors": mp.get("neighbors", WAFFLEIRON_DEFAULTS["neighbors"]),
            "default_num_points": prof.get("num_points", WAFFLEIRON_DEFAULTS["default_num_points"]),
        }
    else:
        wi = raw["waffleiron"]
        emb = raw["embedding"]
        cl = raw["classif"]
        dl = raw.get("dataloader", {})
        cfg = {
            "label": f"WaffleIron  ({os.path.basename(path)})",
            "input_channels": emb["size_input"],
            "feat_channels": wi["nb_channels"],
            "depth": wi["depth"],
            "nb_class": cl["nb_class"],
            "grids_size": wi["grids_size"],
            "neighbors": emb.get("neighbors", WAFFLEIRON_DEFAULTS["neighbors"]),
            "default_num_points": dl.get("max_points", WAFFLEIRON_DEFAULTS["default_num_points"]),
        }
    return cfg


def load_config(path, arch=None):
    """Load a YAML config file and return (arch_type, config_dict)."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if arch is None:
        arch = detect_arch(raw)
    if arch is None:
        raise ValueError(
            f"Cannot auto-detect architecture from {path}. "
            "Use --arch cylinder3d|waffleiron."
        )

    if arch == "cylinder3d":
        return arch, load_cylinder3d_config(path)
    elif arch == "waffleiron":
        return arch, load_waffleiron_config(path)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_cylinder3d(cfg, args):
    np_ = args.num_points or cfg["default_num_points"]
    nv  = args.num_voxels or cfg["default_num_voxels"]

    detail, fea_tot, bb_tot, total, n_levels = cylinder3d_flops(cfg, np_, nv)

    print(f"  Input assumptions:")
    print(f"    Points / scan : {np_:>10,}")
    print(f"    Active voxels : {nv:>10,}")
    print(f"    Grid shape    : {cfg['output_shape']}")
    print(f"    init_size     : {cfg['init_size']}")
    print(f"    num_scales    : {cfg.get('num_scales',1)}")
    print(f"    spconv input C: {cfg['num_input_features'] * cfg.get('num_scales',1)}")

    print(f"\n  Active voxels per encoder level:")
    for i, v in enumerate(n_levels):
        print(f"    Level {i}: {v:>10,}")

    print(f"\n  FLOPs breakdown:")
    print(f"  {'-'*55}")
    for name, flops in detail.items():
        print(f"    {name:<35s} {fmt(flops, 'FLOPs'):>18s}")
    print(f"  {'-'*55}")
    print(f"    {'Feature Generator':<35s} {fmt(fea_tot, 'FLOPs'):>18s}")
    print(f"    {'3-D Backbone':<35s} {fmt(bb_tot, 'FLOPs'):>18s}")
    print(f"  {'='*55}")
    print(f"    {'TOTAL':<35s} {fmt(total, 'FLOPs'):>18s}")

    est_p = cylinder3d_params(cfg)
    print(f"\n  Estimated parameters: {fmt(est_p)}")
    return est_p


def print_waffleiron(cfg, args):
    N = args.num_points or cfg["default_num_points"]

    detail, emb_tot, bb_tot, total = waffleiron_flops(cfg, N)

    print(f"  Input assumptions:")
    print(f"    Points        : {N:>10,}")
    print(f"    Channels      : {cfg['feat_channels']}")
    print(f"    Depth         : {cfg['depth']}")
    print(f"    Grid shapes   : {cfg['grids_size']}")
    print(f"    Neighbors (K) : {cfg.get('neighbors', 16)}")

    print(f"\n  FLOPs breakdown:")
    print(f"  {'-'*55}")
    for name, flops in detail.items():
        print(f"    {name:<35s} {fmt(flops, 'FLOPs'):>18s}")
    print(f"  {'-'*55}")
    print(f"    {'Embedding':<35s} {fmt(emb_tot, 'FLOPs'):>18s}")
    print(f"    {'Backbone':<35s} {fmt(bb_tot, 'FLOPs'):>18s}")
    print(f"  {'='*55}")
    print(f"    {'TOTAL':<35s} {fmt(total, 'FLOPs'):>18s}")

    est_p = waffleiron_params(cfg)
    print(f"\n  Estimated parameters: {fmt(est_p)}")
    return est_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analytical FLOPs estimation for Cylinder3D and WaffleIron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (Cylinder3D or WaffleIron)",
    )
    parser.add_argument(
        "--arch", choices=["cylinder3d", "waffleiron"], default=None,
        help="Force architecture type (auto-detected from YAML if omitted)",
    )
    parser.add_argument(
        "--num-points", type=int, default=None,
        help="Number of input points  (default: model-specific)",
    )
    parser.add_argument(
        "--num-voxels", type=int, default=None,
        help="Number of active voxels at the finest level (Cylinder3D only)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt / .pth checkpoint - used to verify parameter count",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        parser.error(f"Config file not found: {args.config}")

    arch, cfg = load_config(args.config, args.arch)

    section(cfg["label"])

    if arch == "cylinder3d":
        est_p = print_cylinder3d(cfg, args)
    else:
        est_p = print_waffleiron(cfg, args)

    # Optionally verify against a real checkpoint
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"\n  WARNING: checkpoint not found: {args.checkpoint}")
        else:
            try:
                real_p = params_from_checkpoint(args.checkpoint)
                diff = abs(real_p - est_p) / max(real_p, 1) * 100
                print(f"  Checkpoint params : {fmt(real_p)}")
                print(f"  Difference        : {diff:.1f}%")
            except Exception as exc:
                print(f"\n  ERROR loading checkpoint: {exc}")

    print()


if __name__ == "__main__":
    main()
