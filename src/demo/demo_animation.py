from manim import *

class PipelineAnimation(Scene):
    def construct(self):
        self.camera.background_color = "#191926"
        
        left_center = LEFT * 3.5
        
        raw_node = RoundedRectangle(width=5, height=1.5, corner_radius=0.5, color=WHITE).move_to(left_center + UP * 2.5)
        raw_text = Text("Raw Point Cloud").scale(0.6).move_to(raw_node)
        
        nuc_node = RoundedRectangle(width=5, height=1.5, corner_radius=0.5, color=BLUE).move_to(left_center)
        nuc_text = Text("Distilled NUC-Net").scale(0.6).move_to(nuc_node)
        
        alpine_node = RoundedRectangle(width=5, height=1.5, corner_radius=0.5, color=GREEN).move_to(left_center + DOWN * 2.5)
        alpine_text = Text("Alpine Head").scale(0.6).move_to(alpine_node)
        
        arrow1 = Arrow(raw_node.get_bottom(), nuc_node.get_top(), buff=0.1)
        arrow2 = Arrow(nuc_node.get_bottom(), alpine_node.get_top(), buff=0.1)
        
        # t=0 to t=5 (Raw Fullscreen)
        self.wait(5.0)
        
        # t=5 to t=6: Raw Block appears
        self.play(Create(raw_node), Write(raw_text), run_time=1.0)
        
        # t=6 to t=8: Wait for video to shrink
        self.wait(2.0)
        
        # t=8 to t=10: NUC-Net block appears
        self.play(GrowArrow(arrow1), run_time=1.0)
        self.play(Create(nuc_node), Write(nuc_text), run_time=1.0)
        
        # t=10 to t=17: NUC-Net indicate + Semantic Video expands and goes fullscreen
        self.play(Indicate(nuc_node, color=BLUE, scale_factor=1.1), run_time=1.0)
        self.wait(6.0)
        
        # t=17 to t=19: Alpine block appears (video is shrinking into it)
        self.play(GrowArrow(arrow2), run_time=1.0)
        self.play(Create(alpine_node), Write(alpine_text), run_time=1.0)
        
        # t=19 to t=40: Alpine indicate + Video expands and goes fullscreen
        self.play(Indicate(alpine_node, color=GREEN, scale_factor=1.1), run_time=1.0)
        self.wait(20.0)
