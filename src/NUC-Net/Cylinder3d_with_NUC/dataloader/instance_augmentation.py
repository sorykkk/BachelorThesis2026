# -*- coding:utf-8 -*-
# Instance Augmentation for LiDAR semantic segmentation
# Based on instance copy-paste augmentation used in Cylinder3D, RPVNet, PanopticPolarNet
# and cited as [30,31,15] in the NUC-Net paper.

import os
import numpy as np
import pickle
import yaml
from pathlib import Path

# "Thing" classes in SemanticKITTI after label mapping (classes 1-8 are instances):
# 1: car, 2: bicycle, 3: motorcycle, 4: truck, 5: other-vehicle,
# 6: person, 7: bicyclist, 8: motorcyclist
THING_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]


def build_instance_database(data_path, label_mapping_path, split_sequences=None,
                            save_path=None, max_instances_per_class=500):
    """
    Scan training sequences and extract individual instances of "thing" classes.
    Each instance is stored as (xyz_centered, labels, intensity, centroid).
    """
    with open(label_mapping_path, 'r') as f:
        semkittiyaml = yaml.safe_load(f)
    learning_map = semkittiyaml['learning_map']

    if split_sequences is None:
        split_sequences = semkittiyaml['split']['train']

    instance_db = {c: [] for c in THING_CLASSES}

    for seq in split_sequences:
        seq_dir = os.path.join(data_path, str(seq).zfill(2))
        velodyne_dir = os.path.join(seq_dir, 'velodyne')
        label_dir = os.path.join(seq_dir, 'labels')

        if not os.path.isdir(velodyne_dir):
            continue

        scan_files = sorted(os.listdir(velodyne_dir))
        # Subsample scans for efficiency (every 5th frame)
        scan_files = scan_files[::5]

        for scan_file in scan_files:
            scan_path = os.path.join(velodyne_dir, scan_file)
            label_path = os.path.join(label_dir, scan_file.replace('.bin', '.label'))
            if not os.path.exists(label_path):
                continue

            raw_data = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
            xyz = raw_data[:, :3]
            intensity = raw_data[:, 3]

            raw_labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1,))
            sem_labels = raw_labels & 0xFFFF
            inst_labels = raw_labels >> 16

            # Map semantic labels
            mapped_sem = np.vectorize(learning_map.__getitem__)(sem_labels)

            # Extract instances for each thing class
            for cls_id in THING_CLASSES:
                if len(instance_db[cls_id]) >= max_instances_per_class:
                    continue

                cls_mask = (mapped_sem == cls_id)
                if cls_mask.sum() < 10:
                    continue

                cls_inst_labels = inst_labels[cls_mask]
                unique_instances = np.unique(cls_inst_labels)

                for inst_id in unique_instances:
                    if inst_id == 0:
                        continue
                    inst_mask = cls_mask & (inst_labels == inst_id)
                    n_pts = inst_mask.sum()
                    if n_pts < 10 or n_pts > 10000:
                        continue

                    inst_xyz = xyz[inst_mask].copy()
                    inst_int = intensity[inst_mask].copy()
                    centroid = inst_xyz.mean(axis=0)
                    inst_xyz_centered = inst_xyz - centroid

                    instance_db[cls_id].append({
                        'xyz': inst_xyz_centered,
                        'intensity': inst_int,
                        'centroid': centroid,
                        'n_pts': n_pts,
                        'cls': cls_id,
                    })

                    if len(instance_db[cls_id]) >= max_instances_per_class:
                        break

    total = sum(len(v) for v in instance_db.values())
    print(f"[IA] Built instance database: {total} instances total")
    for c in THING_CLASSES:
        print(f"  class {c}: {len(instance_db[c])} instances")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(instance_db, f)
        print(f"[IA] Saved instance database to {save_path}")

    return instance_db


class InstanceAugmentor:
    """
    Instance Augmentation: randomly paste "thing" instances into a scene.
    Following the approach used in Cylinder3D / PanopticPolarNet / RPVNet.
    """

    def __init__(self, instance_db_path, max_paste_per_class=3):
        """
        Args:
            instance_db_path: path to pickle file with instance database
            max_paste_per_class: max number of instances to paste per thing class
        """
        if isinstance(instance_db_path, str):
            with open(instance_db_path, 'rb') as f:
                self.instance_db = pickle.load(f)
        else:
            self.instance_db = instance_db_path

        self.max_paste_per_class = max_paste_per_class
        self.thing_classes = [c for c in THING_CLASSES if len(self.instance_db.get(c, [])) > 0]
        total = sum(len(self.instance_db.get(c, [])) for c in self.thing_classes)
        print(f"[IA] InstanceAugmentor initialized: {total} instances across {len(self.thing_classes)} classes")
        for c in self.thing_classes:
            print(f"  [IA] class {c}: {len(self.instance_db[c])} instances")

        # Verification counters (log once at 500 calls to confirm IA works)
        self._call_count = 0
        self._total_pasted = 0
        self._total_pasted_pts = 0
        self._logged = False

    def augment(self, xyz, labels, intensity=None):
        """
        Paste random instances into the scene.

        Args:
            xyz: (N, 3) point cloud in cartesian coords
            labels: (N, 1) semantic labels
            intensity: (N,) intensity values or None

        Returns:
            aug_xyz, aug_labels, aug_intensity (or None)
        """
        n_orig = len(xyz)
        pasted_count = 0
        new_xyz_list = [xyz]
        new_label_list = [labels]
        new_int_list = [intensity] if intensity is not None else []

        for cls_id in self.thing_classes:
            db_entries = self.instance_db[cls_id]
            if len(db_entries) == 0:
                continue

            # Paste 1 to max_paste_per_class instances of this class
            n_paste = np.random.randint(0, self.max_paste_per_class + 1)
            if n_paste == 0:
                continue

            chosen = np.random.choice(len(db_entries), min(n_paste, len(db_entries)), replace=False)

            for idx in chosen:
                inst = db_entries[idx]
                inst_xyz = inst['xyz'].copy()
                inst_int = inst['intensity'].copy()
                n_pts = inst['n_pts']

                # Random rotation around z-axis
                theta = np.random.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
                inst_xyz = inst_xyz @ rot.T

                # Random placement: pick a random existing point as anchor
                anchor_idx = np.random.randint(0, len(xyz))
                anchor = xyz[anchor_idx].copy()
                # Place instance near the anchor with some offset
                offset = np.array([
                    np.random.uniform(-3, 3),
                    np.random.uniform(-3, 3),
                    0
                ], dtype=np.float32)
                inst_xyz = inst_xyz + anchor + offset

                inst_labels = np.full((n_pts, 1), cls_id, dtype=np.uint8)

                new_xyz_list.append(inst_xyz)
                new_label_list.append(inst_labels)
                if intensity is not None:
                    new_int_list.append(inst_int)
                pasted_count += 1

        # IA verification logging (once at 500 calls)
        self._call_count += 1
        self._total_pasted += pasted_count
        self._total_pasted_pts += sum(len(x) for x in new_xyz_list[1:])
        if not self._logged and self._call_count >= 500:
            self._logged = True
            avg_pasted = self._total_pasted / self._call_count
            avg_pts = self._total_pasted_pts / self._call_count
            print(f"[IA] Verification after {self._call_count} calls: avg {avg_pasted:.1f} instances/scene, "
                  f"avg {avg_pts:.0f} extra pts/scene ({avg_pts / (n_orig + avg_pts) * 100:.1f}% of total)")

        aug_xyz = np.concatenate(new_xyz_list, axis=0)
        aug_labels = np.concatenate(new_label_list, axis=0)
        aug_intensity = np.concatenate(new_int_list, axis=0) if intensity is not None else None

        return aug_xyz, aug_labels, aug_intensity
