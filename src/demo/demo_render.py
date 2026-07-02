import os
import math
import numpy as np
import open3d as o3d
import imageio
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'demo', 'data')
RENDER_DIR = os.path.join(ROOT_DIR, 'demo', 'renders')

os.makedirs(RENDER_DIR, exist_ok=True)

color_map = {
    0: [0, 0, 0], 1: [100, 150, 245], 2: [100, 230, 245], 3: [30, 60, 150], 4: [80, 30, 180],
    5: [100, 80, 250], 6: [255, 30, 30], 7: [255, 40, 200], 8: [150, 30, 90], 9: [255, 0, 255],
    10: [255, 150, 255], 11: [75, 0, 75], 12: [175, 0, 75], 13: [255, 200, 0], 14: [255, 120, 50],
    15: [0, 175, 0], 16: [135, 60, 0], 17: [150, 240, 80], 18: [255, 240, 150], 19: [255, 0, 0],
}

def generate_random_colors(num_colors):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_colors, 3)) / 255.0

def project_3d_to_2d(point, eye, center, up, fov_deg, width, height):
    f = (center - eye) / np.linalg.norm(center - eye)
    u = np.array(up) / np.linalg.norm(up)
    s = np.cross(f, u)
    s_norm = np.linalg.norm(s)
    if s_norm == 0: return None
    s = s / s_norm
    u = np.cross(s, f)
    
    view = np.eye(4)
    view[0, :3] = s; view[1, :3] = u; view[2, :3] = -f
    view[0, 3] = -np.dot(s, eye); view[1, 3] = -np.dot(u, eye); view[2, 3] = np.dot(f, eye)
    
    focal = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    aspect = width / height
    proj = np.zeros((4, 4))
    proj[0, 0] = focal / aspect
    proj[1, 1] = focal
    proj[2, 2] = -(1000 + 0.1) / (1000 - 0.1)
    proj[2, 3] = -2.0 * 1000 * 0.1 / (1000 - 0.1)
    proj[3, 2] = -1.0
    
    p = np.array([point[0], point[1], point[2], 1.0])
    p_view = view @ p
    p_proj = proj @ p_view
    if p_proj[3] <= 0: return None
    
    x = (p_proj[0] / p_proj[3] + 1) * 0.5 * width
    y = (1 - p_proj[1] / p_proj[3]) * 0.5 * height
    return int(x), int(y)

def capture_rotation_video(pcd, video_name, frames=300, fps=30, zoom_target=None, label_points=None):
    video_path = os.path.join(RENDER_DIR, video_name)
    width, height = 1920, 1080
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0
    
    render.scene.add_geometry("pcd", pcd, mat)
    render.scene.set_background([0.1, 0.1, 0.15, 1.0])
    
    bounds = pcd.get_axis_aligned_bounding_box()
    global_center = bounds.get_center()
    max_extent = max(bounds.get_extent())
    
    print(f"Rendering {video_name} with {frames} frames...")
    writer = imageio.get_writer(video_path, fps=fps, macro_block_size=None)
    
    # 480 frames for a full 360 degree rotation (very slow)
    rotation_step = 2 * np.pi / 480 
    start_dist = max_extent * 0.6
    
    for i in range(frames):
        t = i / float(frames - 1) if frames > 1 else 1.0
        smooth_t = t * t * (3 - 2 * t)
        
        if zoom_target is not None:
            # For Alpine, start zooming immediately and hold it
            zoom_factor = min(1.0, t * 2.0) # Reach zoom halfway through
            smooth_z = zoom_factor * zoom_factor * (3 - 2 * zoom_factor)
            current_center = global_center * (1 - smooth_z) + zoom_target * smooth_z
            current_dist = start_dist * (1 - smooth_z) + (max_extent * 0.12) * smooth_z
        else:
            current_center = global_center
            current_dist = start_dist
            
        angle = i * rotation_step
        eye = current_center + np.array([current_dist * np.cos(angle), current_dist * np.sin(angle), current_dist * 0.3])
        up = [0, 0, 1]
        
        render.setup_camera(60.0, current_center, eye, up)
        img = np.asarray(render.render_to_image())
        
        if label_points and zoom_target is not None:
            # fade in labels after zoom starts
            if t > 0.2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                alpha = min(1.0, (t - 0.2) * 5.0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                for point, text, color in label_points:
                    hover_point = point + np.array([0, 0, 1.5])
                    coord = project_3d_to_2d(hover_point, eye, current_center, up, 60.0, width, height)
                    if coord:
                        cv2.putText(img, text, (coord[0], coord[1]), font, 1.0, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(img, text, (coord[0], coord[1]), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        writer.append_data(img)

    writer.close()
    render.scene.remove_geometry("pcd")
    print(f"Saved to {video_path}")

def render_demo():
    print("Loading tensors...")
    xyz = np.load(os.path.join(DATA_DIR, 'input_pc.npy'))
    sem_labels = np.load(os.path.join(DATA_DIR, 'semantic_labels.npy'))
    inst_labels = np.load(os.path.join(DATA_DIR, 'instance_labels.npy'))

    center = np.mean(xyz, axis=0)
    xyz_centered = xyz - center

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_centered)
    
    # 1. Raw Point Cloud
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    z_norm = (xyz[:, 2] - z_min) / (z_max - z_min)
    raw_colors = np.zeros((xyz.shape[0], 3))
    raw_colors[:, 0] = z_norm * 0.8
    raw_colors[:, 1] = z_norm * 0.8
    raw_colors[:, 2] = 0.8 + z_norm * 0.2
    pcd.colors = o3d.utility.Vector3dVector(raw_colors)
    capture_rotation_video(pcd, '01_raw_scene.mp4', frames=300)

    # 2. Semantic Labels
    sem_colors = np.zeros((xyz.shape[0], 3))
    for sem_class, color in color_map.items():
        mask = (sem_labels == sem_class)
        sem_colors[mask] = np.array(color) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(sem_colors)
    capture_rotation_video(pcd, '02_semantic_backbone.mp4', frames=360)

    # 3. Instance Labels (Alpine Head)
    inst_colors = np.zeros((xyz.shape[0], 3))
    unique_inst = np.unique(inst_labels)
    rand_colors = generate_random_colors(len(unique_inst) + 1)
    
    inst_colors[inst_labels == 0] = sem_colors[inst_labels == 0] * 0.3
    
    label_points = []
    zoom_target = None

    for i, inst_id in enumerate(unique_inst):
        if inst_id == 0: continue
        mask = (inst_labels == inst_id)
        inst_colors[mask] = rand_colors[i]
        
        if np.any((sem_labels[mask] == 1)):
            car_pts = xyz_centered[mask]
            if len(car_pts) > 50:
                car_center = np.mean(car_pts, axis=0)
                label_points.append((car_center, f"Inst {inst_id}", rand_colors[i]))

    car_mask = (sem_labels == 1) & (inst_labels > 0)
    if np.any(car_mask):
        u_cars, counts = np.unique(inst_labels[car_mask], return_counts=True)
        target_car_id = u_cars[np.argmax(counts)]
        target_points = xyz_centered[inst_labels == target_car_id]
        zoom_target = np.mean(target_points, axis=0)

    pcd.colors = o3d.utility.Vector3dVector(inst_colors)
    capture_rotation_video(pcd, '03_alpine_instances.mp4', frames=720, zoom_target=zoom_target, label_points=label_points)
    
    print("All renders completed successfully!")

if __name__ == "__main__":
    render_demo()
