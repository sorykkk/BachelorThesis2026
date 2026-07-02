# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Evaluation for WaffleIron on SemanticKITTI
# Saves .label predictions (Alpine-compatible) AND profiles metrics:
#   mIoU, Inference Speed, FPS, VRAM peak, Num Params, FLOPS
#
# USAGE:
# cd experiments/WaffleIron
# python eval_kitti.py \
#   --config ./configs/WaffleIron-48-256__kitti.yaml \
#   --model_load_path ./logs/ckpt_last.pth \
#   --path_dataset ../../data/semantickitti/dataset/ \
#   --result_folder ./predictions_kitti \
#   --phase val \
#   --num_votes 1 \
#   --log_path ./performance_logs

import os
import json
import time
import yaml
import torch
import argparse
import waffleiron
import numpy as np
from tqdm import tqdm
from waffleiron import Segmenter
from datasets import SemanticKITTI, Collate


if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_load_path", type=str, help="Path to model checkpoint to load")
    parser.add_argument(
        "--path_dataset", type=str, help="Path to SemanticKITTI dataset"
    )
    parser.add_argument("--result_folder", type=str, help="Path to where result folder")
    parser.add_argument(
        "--num_votes", type=int, default=1, help="Number of test time augmentations"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--phase", required=True, help="val or test")
    parser.add_argument("--log_path", type=str, default="./performance_logs",
                        help="Directory to save metrics JSON")
    args = parser.parse_args()
    assert args.num_votes % args.batch_size == 0
    os.makedirs(args.result_folder, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # --- Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- SemanticKITTI (from https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py)
    with open("./datasets/semantic-kitti.yaml") as stream:
        semkittiyaml = yaml.safe_load(stream)
    remapdict = semkittiyaml["learning_map_inv"]
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    # --- Dataloader
    tta = args.num_votes > 1
    dataset = SemanticKITTI(
        rootdir=args.path_dataset,
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors=config["embedding"]["neighbors"],
        dim_proj=config["waffleiron"]["dim_proj"],
        grids_shape=config["waffleiron"]["grids_size"],
        fov_xyz=config["waffleiron"]["fov_xyz"],
        phase=args.phase,
        tta=tta,
    )
    if args.num_votes > 1:
        new_list = []
        for f in dataset.im_idx:
            for v in range(args.num_votes):
                new_list.append(f)
        dataset.im_idx = new_list
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )
    args.num_votes = args.num_votes // args.batch_size

    # --- Build network
    net = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"]["drop"],
    )
    net = net.cuda()

    # --- Load weights
    ckpt = torch.load(args.model_load_path, map_location="cuda:0", weights_only=False)
    try:
        net.load_state_dict(ckpt["net"])
    except:
        # If model was trained using DataParallel or DistributedDataParallel
        state_dict = {}
        for key in ckpt["net"].keys():
            state_dict[key[len("module."):]] = ckpt["net"][key]
        net.load_state_dict(state_dict)

    # --- Model complexity metrics (count BEFORE compression)
    nb_params_M = sum(p.numel() for p in net.parameters()) / 1e6
    nb_trainable_M = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    print(f"Num Params: {nb_params_M:.2f}M (trainable: {nb_trainable_M:.2f}M)")

    # --- Compress model (folds BN into Conv for faster inference)
    net.compress()
    net.eval()

    # --- Re-activate droppath if voting
    if tta:
        for m in net.modules():
            if isinstance(m, waffleiron.backbone.DropPath):
                m.train()

    # --- Evaluation
    nb_class = config["classif"]["nb_class"]
    confusion_matrix = np.zeros((nb_class, nb_class), dtype=np.int64)
    frame_times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    id_vote = 0
    for it, batch in enumerate(
        tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Reset vote
        if id_vote == 0:
            vote = None

        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
        labels = batch["labels_orig"].cuda(non_blocking=True)
        batch["upsample"] = [up.cuda(non_blocking=True) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction
        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                out = net(*net_inputs)
                torch.cuda.synchronize()
                frame_time = time.perf_counter() - t_start
                frame_times.append(frame_time)

                for b in range(out.shape[0]):
                    temp = out[b, :, batch["upsample"][b]].T
                    if vote is None:
                        vote = torch.softmax(temp, dim=1)
                    else:
                        vote += torch.softmax(temp, dim=1)
        id_vote += 1

        # Save prediction
        if id_vote == args.num_votes:
            # Convert label
            pred_label = (
                vote.max(1)[1] + 1
            )  # Shift by 1 because of ignore_label at index 0
            label = pred_label.cpu().numpy().reshape(-1).astype(np.uint32)
            upper_half = label >> 16  # get upper half for instances
            lower_half = label & 0xFFFF  # get lower half for semantics
            lower_half = remap_lut[lower_half]  # do the remapping of semantics
            label = (upper_half << 16) + lower_half  # reconstruct full label
            label = label.astype(np.uint32)

            # Compute confusion matrix for mIoU (val phase only)
            if args.phase == "val":
                gt = labels.cpu().numpy().reshape(-1)
                pred_for_iou = (vote.max(1)[1]).cpu().numpy().reshape(-1)
                valid = gt != 255
                if valid.any():
                    np.add.at(
                        confusion_matrix,
                        (gt[valid], pred_for_iou[valid]),
                        1,
                    )

            # Save result
            assert batch["filename"][0] == batch["filename"][-1]
            label_file = batch["filename"][0][
                len(os.path.join(dataset.rootdir, "dataset/")):
            ]
            label_file = label_file.replace("velodyne", "predictions")[:-3] + "label"
            label_file = os.path.join(args.result_folder, label_file)
            os.makedirs(os.path.split(label_file)[0], exist_ok=True)
            label.tofile(label_file)
            # Reset count of votes
            id_vote = 0

    # --- Compute and print metrics
    # mIoU
    if args.phase == "val":
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.diag(confusion_matrix) / (
                confusion_matrix.sum(1) + confusion_matrix.sum(0) - np.diag(confusion_matrix)
            )
        miou = float(np.nanmean(ious)) * 100.0
        print(f"\nmIoU: {miou:.2f}%")
        for i, name in enumerate(SemanticKITTI.CLASS_NAME):
            print(f"  {name:<20s} {ious[i]*100:.2f}%")
    else:
        miou = 0.0
        ious = np.zeros(nb_class)

    # Inference timing - normalize per frame
    ft = np.array(frame_times)
    ft_stats = ft[1:] if len(ft) > 10 else ft  # skip warmup
    mean_batch_ms = float(np.mean(ft_stats)) * 1000.0
    mean_frame_ms = mean_batch_ms / args.batch_size
    fps = 1000.0 / mean_frame_ms if mean_frame_ms > 0 else 0.0
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print(f"\n{'='*60}")
    print(f"  WaffleIron Metrics Summary - SemanticKITTI ({args.phase})")
    print(f"{'='*60}")
    print(f"  Num Params (M):              {nb_params_M:.2f}")
    if args.phase == "val":
        print(f"  mIoU:                        {miou:.2f}%")
    print(f"  Batch size:                  {args.batch_size}")
    print(f"  Speed (per batch, ms):       {mean_batch_ms:.2f}")
    print(f"  Speed (per frame, ms):       {mean_frame_ms:.2f}")
    print(f"  FPS:                         {fps:.2f}")
    print(f"  VRAM Peak (MB):              {vram_peak:.1f}")
    print(f"  Total batches:               {len(ft)}")
    print(f"{'='*60}\n")

    # Save single metrics.json (Alpine-compatible)
    metrics = {
        "num_params_M": nb_params_M,
        "mIoU": miou / 100.0,
        "batch_size": args.batch_size,
        "inference_speed_ms": mean_frame_ms,
        "fps": fps,
        "vram_peak_mb": vram_peak,
        "total_frames": len(ft) * args.batch_size,
        "num_votes": args.num_votes * args.batch_size,
        "phase": args.phase,
    }
    if args.phase == "val":
        for i, name in enumerate(SemanticKITTI.CLASS_NAME):
            metrics[f"iou/{name}"] = float(ious[i])
    metrics_path = os.path.join(args.log_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Predictions saved to {args.result_folder}")
