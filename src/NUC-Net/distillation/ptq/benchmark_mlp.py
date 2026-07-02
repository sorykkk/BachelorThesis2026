import sys
import time
import torch
from pathlib import Path

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_DISTILL_ROOT = _SCRIPT_DIR.parent
_NUCNET_DIR = _DISTILL_ROOT.parent
_NUCNET_ROOT = _NUCNET_DIR / "Cylinder3d_with_NUC"

sys.path.insert(0, str(_DISTILL_ROOT))
sys.path.insert(0, str(_NUCNET_ROOT))

from network.cylinder_fea_generator import cylinder_fea

def benchmark():
    device = torch.device('cpu')
    
    # Instantiate MLP feature generator (Student size)
    model = cylinder_fea(
        grid_size=[120, 360, 32],
        fea_dim=9,
        out_pt_fea_dim=256,
        fea_compre=16,
        num_scales=2
    ).to(device)
    model.eval()

    # Dummy data (simulate ~100k points in a typical SemanticKITTI scan)
    N_points = 100000
    pt_fea = [torch.randn(N_points, 9, device=device)]
    xy_ind = [torch.randint(0, 32, (N_points, 3), device=device)]
    xy_ind_ms = [torch.randint(0, 16, (N_points, 3), device=device)]

    def measure(m, name):
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                m(pt_fea, xy_ind, xy_ind_ms)
        
        # Benchmark
        start = time.time()
        iters = 50
        with torch.no_grad():
            for _ in range(iters):
                m(pt_fea, xy_ind, xy_ind_ms)
        end = time.time()
        
        avg_ms = ((end - start) / iters) * 1000
        print(f"[{name}] Average Latency: {avg_ms:.2f} ms")
        return avg_ms

    print("="*50)
    print("Isolated MLP CPU Benchmark (100k points)")
    print("="*50)
    
    # 1. FP32 CPU
    print("1. Benchmarking unquantized (FP32) on CPU...")
    fp32_ms = measure(model, "FP32 CPU")

    # 2. Quantize to INT8
    print("\n2. Quantizing to INT8...")
    q_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    print("3. Benchmarking quantized (INT8) on CPU...")
    int8_ms = measure(q_model, "INT8 CPU")

    print("\n" + "="*50)
    if int8_ms < fp32_ms:
        print(f"Result: INT8 is {fp32_ms / int8_ms:.2f}x FASTER than FP32 on CPU!")
    else:
        print(f"Result: INT8 is {int8_ms / fp32_ms:.2f}x SLOWER than FP32 on CPU.")
    print("="*50)

if __name__ == '__main__':
    benchmark()