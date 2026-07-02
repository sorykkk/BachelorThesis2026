import os
from moviepy import VideoFileClip, CompositeVideoClip

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_DIR = os.path.join(ROOT_DIR, 'demo', 'renders')

def find_manim_video():
    project_root = os.path.dirname(ROOT_DIR)
    base_dir = os.path.join(project_root, 'media', 'videos', 'demo_animation')
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "PipelineAnimation.mp4":
                return os.path.join(root, file)
    raise FileNotFoundError("Manim animation video not found.")

def interpolate(t, keyframes):
    if t <= keyframes[0][0]: return keyframes[0][1]
    if t >= keyframes[-1][0]: return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        t0, v0 = keyframes[i]
        t1, v1 = keyframes[i+1]
        if t0 <= t <= t1:
            ratio = (t - t0) / (t1 - t0)
            smooth_ratio = ratio * ratio * (3 - 2 * ratio)
            return v0 + (v1 - v0) * smooth_ratio
    return keyframes[-1][1]

def make_dynamic_clip(clip, keyframes):
    def scale_func(t):
        s = interpolate(t, [(k[0], k[1]) for k in keyframes])
        return max(0.001, s)
        
    def pos_func(t):
        s = interpolate(t, [(k[0], k[1]) for k in keyframes])
        cx = interpolate(t, [(k[0], k[2]) for k in keyframes])
        cy = interpolate(t, [(k[0], k[3]) for k in keyframes])
        w = 1920 * s
        h = 1080 * s
        return (cx - w/2, cy - h/2)
        
    # Resize and position dynamically!
    return clip.resized(scale_func).with_position(pos_func)

def composite_videos():
    print("Loading Manim animation...")
    manim_path = find_manim_video()
    manim_clip = VideoFileClip(manim_path)
    
    print("Loading Open3D videos...")
    raw_clip = VideoFileClip(os.path.join(RENDER_DIR, '01_raw_scene.mp4'))
    sem_clip = VideoFileClip(os.path.join(RENDER_DIR, '02_semantic_backbone.mp4'))
    alp_clip = VideoFileClip(os.path.join(RENDER_DIR, '03_alpine_instances.mp4'))
    
    # Keyframes: (local_time, scale, center_x, center_y)
    
    # 1. Raw Clip (plays composite t=0 to t=7) -> duration 7s
    raw_keys = [
        (0.0, 1.0, 960, 540),
        (5.0, 1.0, 960, 540),
        (7.0, 0.0, 488, 203)
    ]
    raw_dyn = make_dynamic_clip(raw_clip.subclipped(0, 7.0), raw_keys).with_start(0.0)
    
    # 2. Semantic Clip (plays composite t=7 to t=19) -> duration 12s
    sem_keys = [
        (0.0, 0.0, 488, 540),
        (3.0, 0.0, 488, 540),
        (5.0, 0.45, 1400, 540),
        (7.0, 1.0, 960, 540),
        (10.0, 1.0, 960, 540),
        (12.0, 0.0, 488, 878)
    ]
    sem_dyn = make_dynamic_clip(sem_clip.subclipped(0, 12.0), sem_keys).with_start(7.0)
    
    # 3. Alpine Clip (plays composite t=19 to t=40) -> duration 21s
    alp_keys = [
        (0.0, 0.0, 488, 878),
        (3.0, 0.45, 1400, 540),
        (9.0, 0.45, 1400, 540),
        (13.0, 1.0, 960, 540),
        (21.0, 1.0, 960, 540)
    ]
    alp_dyn = make_dynamic_clip(alp_clip.subclipped(0, 21.0), alp_keys).with_start(19.0)
    
    print("Compositing videos...")
    final_video = CompositeVideoClip([
        manim_clip,
        raw_dyn,
        sem_dyn,
        alp_dyn
    ], size=(1920, 1080))
    
    # Trim to exact 40 seconds
    final_video = final_video.subclipped(0, 40.0)
    
    output_path = os.path.join(RENDER_DIR, 'final_demo.mp4')
    print(f"Writing final composite to {output_path}...")
    final_video.write_videofile(output_path, fps=30, codec='libx264')
    print("Done!")

if __name__ == "__main__":
    composite_videos()
if __name__ == "__main__":
    composite_videos()
