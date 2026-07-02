# Instructions & commands to run

---

## WaffleIron 
### Train backbone (or use pretrained weights)
```
cd WaffleIron
python launch_train.py \
    --dataset semantic_kitti \
    --path_dataset ../../data/semantickitti/ \
    --model_save_path ./model_save_dir/WaffleIron-48-256__kitti/ \
    --log_path ./logs/WaffleIron-48-256__kitti/ \
    --config ./configs/WaffleIron-48-256__kitti.yaml \
    --fp16 \
    --gpu 0 \
    --restart  # Used only when restarting training from last checkpoint
```

### Generate predictions
```
cd WaffleIron
python eval_kitti.py \
    --config ./configs/WaffleIron-48-256__kitti.yaml \
    --model_load_path ./model_save_dir/WaffleIron-48-256__kitti/ckpt_best.pth \
    --path_dataset ../../data/semantickitti/ \
    --result_folder ./predictions_kitti \
    --log_path ./logs/WaffleIron-48-256__kitti/ \
    --phase val \
    --num_votes 1
```

---

## NUC-Net
### Train backbone
```
cd NUC-Net/Cylinder3d_with_NUC
python build_instance_db.py \
    --data_path ../../../data/semantickitti/dataset/sequences/
python train_cyl_sem.py \
    -y config/semantickitti.yaml
```

### Evaluate backbone
```
cd NUC-Net/Cylinder3d_with_NUC
python eval_cyl_sem.py \
    -y config/semantickitti.yaml -p ./predictions \
    -m ./model_save_dir/model_full_ss.pt
```

---

## Distilled NUC-Net
### Train backbone
```
cd NUC-Net/distillation
python training/train_distill.py \
    -y config/distill_semantickitti.yaml \
    # --resume  # Only when resume from checkpoint needed
```

### Evaluate backbone
```
cd NUC-Net/distillation
python evaluation/eval_student.py  \
    -y config/distill_semantickitti.yaml \
    -m ./checkpoints/student_ss_best.pt \
    -p ./predictions \
    --amp
    # --compare_teacher                 # For teacher comparison
```

---

## Alpine head panoptic evaluation

### For WaffleIron
```
cd Alpine
python alpine_semantickitti.py \
    --path_to_files ../WaffleIron/predictions_kitti/ \
    --path_dataset ../../data/semantickitti/dataset/ \
    --split \
    --backbone_metrics_json ../WaffleIron/logs/WaffleIron-48-256__kitti/metrics.json # if file not found, put instead of metrics.json the file that contains the needed metrics \ 
    --log_path ./performance_logs/
```

### For NUC-Net
```
cd Alpine
python alpine_semantickitti.py \
    --path_to_files ../NUC-Net/Cylinder3d_with_NUC/predictions/ \
    --path_dataset ../../data/semantickitti/dataset/ \
    --split \
    --backbone_metrics_json ../NUC-Net/Cylinder3d_with_NUC/logs/metrics.json \
    --log_path ./performance_logs/
```
