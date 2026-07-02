"""
Post-Training Quantization (PTQ) for the NUC-Net Student Model.

This script applies PyTorch Dynamic Quantization to the trained FP32 student model.
Since 3D sparse convolutions (`spconv`) lack native INT8 GEMM support on most
consumer GPUs, this script targets the dense multi-layer perceptrons (MLPs) 
inside the cylinder feature generator.

The MLPs process millions of raw points before they are voxelized. By quantizing
the Linear layers to INT8, we significantly reduce the memory bandwidth and 
computation required during the initial feature extraction phase, while leaving
the sensitive 3D backbone in FP32 (or FP16 during inference) to maintain high mIoU.

Usage:
    cd src/NUC-Net/distillation
    python ptq/quantize_student.py -y config/distill_semantickitti.yaml -m checkpoints/student_ss_best.pt
"""

import os
import sys
import argparse
import torch
import torch.quantization
from pathlib import Path

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_DISTILL_ROOT = _SCRIPT_DIR.parent
_NUCNET_DIR = _DISTILL_ROOT.parent
_NUCNET_ROOT = _NUCNET_DIR / "Cylinder3d_with_NUC"

sys.path.insert(0, str(_DISTILL_ROOT))
sys.path.insert(0, str(_NUCNET_ROOT))
_ORIGINAL_CWD = Path.cwd()
os.chdir(str(_NUCNET_DIR))

from models.student import build_student

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # PTQ is typically performed on CPU first, then saved or traced
    device = torch.device('cpu')
    config = load_config(args.config_path)
    student_config = config['student_params']

    print("\n" + "=" * 60)
    print("1. Building FP32 Student Model")
    print("=" * 60)
    student_model = build_student(student_config, device=str(device))
    
    # Load FP32 trained weights
    ckpt_path = args.model_path
    if not os.path.exists(ckpt_path):
        print(f"[FATAL ERROR] Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    state_dict = torch.load(ckpt_path, map_location=device)
    if 'student_state_dict' in state_dict:
        state_dict = state_dict['student_state_dict']
    
    student_model.load_state_dict(state_dict, strict=False)
    student_model.eval()
    
    fp32_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"[Loaded] FP32 Student Model ({fp32_size:.2f} MB)")

    print("\n" + "=" * 60)
    print("2. Applying Dynamic INT8 Quantization")
    print("=" * 60)
    # Apply dynamic quantization to all nn.Linear layers in the model.
    # This targets the PointNet MLPs in the cylinder feature generator.
    quantized_student = torch.quantization.quantize_dynamic(
        student_model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
    
    print("[Success] Model layers dynamically quantized to INT8.")
    
    # Print the network structure to show the replaced layers
    # (nn.Linear -> nn.quantized.dynamic.modules.linear.Linear)
    has_quantized = False
    for name, module in quantized_student.named_modules():
        if 'quantized' in str(type(module)):
            has_quantized = True
            print(f"  - Quantized Layer: {name} -> {type(module).__name__}")
            
    if not has_quantized:
        print("  - Note: No nn.Linear layers were found to quantize. If your MLP uses nn.Conv1d instead,")
        print("          you may need Static Quantization instead of Dynamic Quantization.")

    print("\n" + "=" * 60)
    print("3. Saving Quantized Model")
    print("=" * 60)
    
    # Save the quantized model
    save_dir = os.path.join(str(_DISTILL_ROOT), "ptq", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "student_int8_dynamic.pt")
    
    # Save state dict
    torch.save({'student_state_dict': quantized_student.state_dict()}, save_path)
    
    int8_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[Saved] Quantized Student Model to {save_path}")
    print(f"[Stats] Size Reduction: {fp32_size:.2f} MB -> {int8_size:.2f} MB")
    print("=" * 60)
    print("\nNext Steps:")
    print("To evaluate this quantized model on CPU, you can modify eval_student.py")
    print("to load the dynamic quantized architecture before loading these INT8 weights.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTQ INT8 Quantization for NUC-Net Student')
    parser.add_argument('-y', '--config_path', default='config/distill_semantickitti.yaml')
    parser.add_argument('-m', '--model_path', required=True, help='Path to FP32 student checkpoint')
    args = parser.parse_args()
    
    args.config_path = str((_ORIGINAL_CWD / args.config_path).resolve())
    args.model_path = str((_ORIGINAL_CWD / args.model_path).resolve())
    main(args)
