#!/usr/bin/env python3
"""
Build instance database for Instance Augmentation (IA).
Run this ONCE before training:
    python build_instance_db.py --data_path ../../data/semantickitti/dataset/sequences/
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader.instance_augmentation import build_instance_database


def main():
    parser = argparse.ArgumentParser(description='Build instance database for IA')
    parser.add_argument('--data_path', type=str,
                        default='../../data/semantickitti/dataset/sequences/',
                        help='Path to SemanticKITTI sequences directory')
    parser.add_argument('--label_mapping', type=str,
                        default='./config/label_mapping/semantic-kitti.yaml',
                        help='Path to label mapping YAML')
    parser.add_argument('--save_path', type=str,
                        default='./instance_db/semantickitti_instance_db.pkl',
                        help='Where to save the instance database')
    parser.add_argument('--max_per_class', type=int, default=500,
                        help='Max instances per class')
    args = parser.parse_args()

    build_instance_database(
        data_path=args.data_path,
        label_mapping_path=args.label_mapping,
        save_path=args.save_path,
        max_instances_per_class=args.max_per_class,
    )


if __name__ == '__main__':
    main()
