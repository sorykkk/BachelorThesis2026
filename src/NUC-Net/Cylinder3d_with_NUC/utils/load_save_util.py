# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


def load_checkpoint(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            match_size += 1
            part_load[k] = value
        else:
            assert len(value.shape) == 1 or len(value.shape) == 5
            if len(value.shape) == 1:
                c = value.shape[0]
                cc = my_model_dict[k].shape[0] - c #int(c*0.5)
                if cc <= c:
                    value = torch.cat([value, value[:cc]], dim=0)
                else:
                    value = torch.cat([value, value, value[:(cc-c)]], dim=0)
            else:
                _, _, _, c1, c2 = value.shape
                cc1 = my_model_dict[k].shape[3] - c1 #int(c1*0.5)
                cc2 = my_model_dict[k].shape[4] - c2 #int(c2*0.5)
                if cc1 > 0 and cc1 <= c1:
                    value1 = torch.cat([value, value[:, :, :, :cc1, :]], dim=3) 
                elif cc1 > c1:
                    value1 = torch.cat([value, value, value[:, :, :, :(cc1-c1), :]], dim=3) 
                else:
                    value1 = value
                if cc2 > 0 and cc2 <= c2:
                    value = torch.cat([value1, value1[:, :, :, :, :cc2]], dim=4) 
                elif cc2 > c2:
                    value = torch.cat([value1, value1, value1[:, :, :, :, :(cc2-c2)]], dim=4) 
                else:
                    value = value1
            nomatch_size += 1
            part_load[k] = value
            assert my_model_dict[k].shape == value.shape
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model

def load_checkpoint_old(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        else:
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model

def save_full_checkpoint(path, model, optimizer, epoch, global_iter, best_val_miou):
    """Save a full training checkpoint (model, optimizer, epoch, iter, best mIoU)."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_iter': global_iter,
        'best_val_miou': best_val_miou,
    }, path)
    print(f"[checkpoint] Saved full checkpoint to {path} (epoch={epoch}, iter={global_iter}, best_miou={best_val_miou:.3f})")


def load_full_checkpoint(path, model, optimizer=None, device='cuda:0'):
    """Load a full training checkpoint. Returns (model, optimizer, epoch, global_iter, best_val_miou).
    If the file is a legacy weights-only checkpoint, loads weights and returns zeros for the rest."""
    checkpoint = torch.load(path, map_location=device)

    # Detect if this is a full checkpoint or a legacy state_dict-only file
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        global_iter = checkpoint.get('global_iter', 0)
        best_val_miou = checkpoint.get('best_val_miou', 0.0)
        print(f"[checkpoint] Resumed full checkpoint from {path} (epoch={epoch}, iter={global_iter}, best_miou={best_val_miou:.3f})")
        return model, optimizer, epoch, global_iter, best_val_miou
    else:
        # Legacy weights-only checkpoint - use existing load_checkpoint logic
        model = load_checkpoint(path, model)
        print(f"[checkpoint] Loaded legacy weights-only checkpoint from {path}")
        return model, optimizer, 0, 0, 0.0


def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
