"""
Metrics profiler for Alpine pipeline.
Provides: InferenceProfiler, count_parameters, save_metrics_json,
          load_backbone_stats, collect_all_metrics, print_metrics_table,
          collect_backbone_metrics, print_backbone_metrics_table.
"""

import time
import json
import numpy as np


class InferenceProfiler:
    """Tracks per-frame inference timing, throughput, and VRAM usage."""

    def __init__(self):
        self.frame_times = []
        self._start_time = None
        self._vram_samples = []
        self._gpu_available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def start_frame(self):
        """Call before processing a frame."""
        if self._gpu_available:
            import torch
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def end_frame(self):
        """Call after processing a frame. Returns elapsed time in seconds."""
        if self._gpu_available:
            import torch
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_time
        self.frame_times.append(elapsed)
        if self._gpu_available:
            import torch
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self._vram_samples.append(vram_mb)
        return elapsed

    def get_stats(self):
        """Return profiling statistics."""
        times = np.array(self.frame_times)
        if len(times) == 0:
            return {
                "inference_speed_ms": 0.0,
                "fps": 0.0,
                "total_frames": 0,
                "total_time_s": 0.0,
                "vram_peak_mb": 0.0,
            }

        if len(times) > 10:
            times_for_stats = times[1:]
        else:
            times_for_stats = times

        total_time = float(np.sum(times))
        mean_time = float(np.mean(times_for_stats))
        fps = 1.0 / mean_time if mean_time > 0 else 0.0
        vram_peak = float(max(self._vram_samples)) if self._vram_samples else 0.0

        return {
            "inference_speed_ms": mean_time * 1000.0,
            "fps": fps,
            "total_frames": len(times),
            "total_time_s": total_time,
            "vram_peak_mb": vram_peak,
        }


def count_parameters(model):
    """Count total and trainable parameters (in millions)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params_M": total / 1e6,
        "trainable_params_M": trainable / 1e6,
    }


def save_metrics_json(metrics, filepath):
    """Save metrics dict to JSON, converting numpy types."""
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Metrics saved to {filepath}")


def load_backbone_stats(metrics_json_path):
    """Load backbone stats from a metrics.json file produced by eval.

    Parameters
    ----------
    metrics_json_path : str
        Full path to the backbone's metrics.json file.

    Returns
    -------
    dict with keys prefixed by ``backbone_``.
    Returns an empty dict when the file is missing.
    """
    import os
    stats = {}

    if not os.path.isfile(metrics_json_path):
        return stats

    with open(metrics_json_path) as f:
        data = json.load(f)

    stats["backbone_num_params_M"] = data.get("num_params_M", 0.0)

    batch_size = data.get("batch_size", 1)
    stats["backbone_inference_speed_ms"] = data.get("inference_speed_ms", 0.0)
    stats["backbone_fps"] = data.get("fps", 0.0)
    stats["backbone_batch_size"] = batch_size
    stats["backbone_vram_peak_mb"] = data.get("vram_peak_mb", 0.0)

    if "mIoU" in data:
        stats["backbone_mIoU"] = data["mIoU"]

    return stats


def collect_all_metrics(
    panoptic_results,
    alpine_profiler_stats=None,
    backbone_stats=None,
):
    """Merge backbone stats, Alpine profiling, and panoptic quality into one dict."""
    metrics = {}

    if backbone_stats:
        metrics["backbone_num_params_M"] = backbone_stats.get("backbone_num_params_M", 0.0)

    for key in ["PQ", "PQ_dagger", "PQ_things", "PQ_stuff",
                 "RQ", "RQ_things", "RQ_stuff",
                 "SQ", "SQ_things", "SQ_stuff",
                 "mAP", "mIoU"]:
        metrics[key] = panoptic_results.get(key, 0.0)

    if backbone_stats:
        metrics["backbone_inference_speed_ms"] = backbone_stats.get("backbone_inference_speed_ms", 0.0)
        metrics["backbone_fps"] = backbone_stats.get("backbone_fps", 0.0)
        metrics["backbone_vram_peak_mb"] = backbone_stats.get("backbone_vram_peak_mb", 0.0)
        if "backbone_mIoU" in backbone_stats:
            metrics["backbone_mIoU"] = backbone_stats["backbone_mIoU"]

    if alpine_profiler_stats:
        metrics["alpine_inference_speed_ms"] = alpine_profiler_stats.get("inference_speed_ms", 0.0)
        metrics["alpine_fps"] = alpine_profiler_stats.get("fps", 0.0)

    bb_ms = metrics.get("backbone_inference_speed_ms", 0.0)
    al_ms = metrics.get("alpine_inference_speed_ms", 0.0)
    if bb_ms > 0 or al_ms > 0:
        total_ms = bb_ms + al_ms
        metrics["total_pipeline_latency_ms"] = total_ms
        metrics["total_pipeline_fps"] = 1000.0 / total_ms if total_ms > 0 else 0.0

    return metrics


def print_metrics_table(metrics, dataset_name=""):
    """Print metrics in a formatted table."""
    header = f"=== Metrics Summary{' - ' + dataset_name if dataset_name else ''} ==="
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    sections = [
        ("Model Complexity (Backbone)", [
            ("Num Params (M)", "backbone_num_params_M", ".2f"),
        ]),
        ("Panoptic Quality", [
            ("PQ", "PQ", ".4f"),
            ("PQ\u2020 (dagger)", "PQ_dagger", ".4f"),
            ("PQ things", "PQ_things", ".4f"),
            ("PQ stuff", "PQ_stuff", ".4f"),
        ]),
        ("Recognition Quality", [
            ("RQ", "RQ", ".4f"),
            ("RQ things", "RQ_things", ".4f"),
            ("RQ stuff", "RQ_stuff", ".4f"),
        ]),
        ("Segmentation Quality", [
            ("SQ", "SQ", ".4f"),
            ("SQ things", "SQ_things", ".4f"),
            ("SQ stuff", "SQ_stuff", ".4f"),
        ]),
        ("Other Quality Metrics", [
            ("mAP", "mAP", ".4f"),
            ("mIoU (Alpine)", "mIoU", ".4f"),
            ("mIoU (Backbone)", "backbone_mIoU", ".4f"),
        ]),
        ("Backbone Inference (GPU forward pass, per frame)", [
            ("Inference Speed (ms/frame)", "backbone_inference_speed_ms", ".2f"),
            ("FPS (frames/s)", "backbone_fps", ".2f"),
            ("VRAM Peak (MB)", "backbone_vram_peak_mb", ".1f"),
        ]),
        ("Alpine Clustering (CPU post-processing, per frame)", [
            ("Inference Speed (ms/frame)", "alpine_inference_speed_ms", ".2f"),
            ("FPS (frames/s)", "alpine_fps", ".2f"),
        ]),
        ("Total Pipeline (Backbone + Alpine)", [
            ("Total Latency (ms/frame)", "total_pipeline_latency_ms", ".2f"),
            ("FPS (frames/s)", "total_pipeline_fps", ".2f"),
        ]),
    ]

    for section_name, entries in sections:
        if not any(metrics.get(key) not in (None, "N/A") for _, key, _ in entries):
            continue
        print(f"\n  {section_name}:")
        for label, key, fmt in entries:
            val = metrics.get(key, "N/A")
            if isinstance(val, (int, float)):
                print(f"    {label:<30s} {val:{fmt}}")
            else:
                print(f"    {label:<30s} {val}")
    print()


def collect_backbone_metrics(profiler_stats, model_stats, miou,
                             per_class_iou=None, class_names=None):
    """Aggregate backbone-only metrics into a single dict."""
    metrics = {}
    metrics["num_params_M"] = model_stats.get("total_params_M", 0.0)
    metrics["mIoU"] = miou
    if per_class_iou is not None and class_names is not None:
        for name, iou_val in zip(class_names, per_class_iou):
            metrics[f"iou/{name}"] = float(iou_val)
    metrics["inference_speed_ms"] = profiler_stats.get("inference_speed_ms", 0.0)
    metrics["fps"] = profiler_stats.get("fps", 0.0)
    metrics["vram_peak_mb"] = profiler_stats.get("vram_peak_mb", 0.0)
    metrics["total_frames"] = profiler_stats.get("total_frames", 0)
    metrics["total_time_s"] = profiler_stats.get("total_time_s", 0.0)
    return metrics


def print_backbone_metrics_table(metrics, dataset_name="", model_name=""):
    """Print backbone-only metrics table."""
    title = model_name or "Backbone"
    header = f"=== {title} Metrics Summary{' - ' + dataset_name if dataset_name else ''} ==="
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    sections = [
        ("Model Complexity", [
            ("Num Params (M)", "num_params_M", ".2f"),
        ]),
        ("Semantic Segmentation Quality", [
            ("mIoU", "mIoU", ".4f"),
        ]),
        ("Inference Performance (per frame)", [
            ("Inference Speed (ms/frame)", "inference_speed_ms", ".2f"),
            ("FPS (frames/s)", "fps", ".2f"),
            ("VRAM Peak (MB)", "vram_peak_mb", ".1f"),
            ("Total frames", "total_frames", "d"),
            ("Total time (s)", "total_time_s", ".2f"),
        ]),
    ]

    for section_name, entries in sections:
        has_vals = any(metrics.get(key) not in (None, "N/A", 0, 0.0) for _, key, _ in entries)
        if not has_vals:
            continue
        print(f"\n  {section_name}:")
        for label, key, fmt in entries:
            val = metrics.get(key, "N/A")
            if isinstance(val, (int, float)):
                print(f"    {label:<35s} {val:{fmt}}")
            else:
                print(f"    {label:<35s} {val}")

    iou_keys = [k for k in metrics if k.startswith("iou/")]
    if iou_keys:
        print(f"\n  Per-class IoU:")
        for k in sorted(iou_keys):
            class_name = k.split("/", 1)[1]
            print(f"    {class_name:<35s} {metrics[k]*100:.2f}%")
    print()
