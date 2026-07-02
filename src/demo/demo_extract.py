import os
import sys
import numpy as np

# Add Alpine path to sys.path so we can import it
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPINE_DIR = os.path.join(ROOT_DIR, 'experiments', 'Alpine')
sys.path.append(ALPINE_DIR)

from alpine import Alpine

def extract_demo_tensors():
    print("Extracting NUC-Net (distilled) and Alpine intermediate tensors...")
    
    # Configuration from alpine_semantickitti.py
    THING_CLASSES = [1,2,3,4,5,6,7,8]
    BBOX_WEB = {
        1: [4.4, 1.8], 2: [1.75, 0.61], 3: [2.2, 0.95], 
        4: [10, 3], 5: [10, 3], 6: [0.94, 0.94], 
        7: [1.75, 0.61], 8: [2.2, 0.95]
    }
    
    mapper = {
        0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 
        30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 
        51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 
        99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
    }

    # Paths
    # Using sequence 08 (Validation), frame 000000
    bin_path = os.path.join(ROOT_DIR, 'data', 'semantickitti', 'dataset', 'sequences', '08', 'velodyne', '000000.bin')
    label_path = os.path.join(ROOT_DIR, 'NUC-Net', 'distillation', 'predictions', 'sequences', '08', 'predictions', '000000.label')
    save_dir = os.path.join(ROOT_DIR, 'demo', 'data')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Point cloud file not found: {bin_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Distilled NUC-Net prediction file not found: {label_path}")

    # Load Raw Point Cloud
    print("1. Loading raw point cloud...")
    pc = np.fromfile(bin_path, dtype=np.float32)
    pc = pc.reshape((-1, 4))
    
    # Save input pc (only x, y, z for visualization)
    xyz = pc[:, :3]
    np.save(os.path.join(save_dir, 'input_pc.npy'), xyz)

    # Load Semantic Backbone Output (Distilled NUC-Net)
    print("2. Loading semantic predictions from Distilled NUC-Net...")
    pred = np.fromfile(label_path, dtype=np.uint32)
    pred = pred & 0xFFFF  # Remove high 16 digits
    
    # Map raw KITTI ids to semantic classes 0-19 used by Alpine
    sem_pred = np.vectorize(mapper.__getitem__)(pred).astype(np.int32)
    
    # Save semantic labels
    np.save(os.path.join(save_dir, 'semantic_labels.npy'), sem_pred)

    # Run Alpine Head
    print("3. Running Alpine Instance Clustering Head...")
    alpine = Alpine(THING_CLASSES, BBOX_WEB, k=32, split=False, margin=1.3)
    
    # alpine.fit_predict expects pc with x, y in the first two columns
    inst_pred = alpine.fit_predict(pc[:, :2], sem_pred)
    
    # Save instance labels
    np.save(os.path.join(save_dir, 'instance_labels.npy'), inst_pred)
    
    print("Extraction complete! Tensors saved to src/demo/data/")

if __name__ == "__main__":
    extract_demo_tensors()
