from manim import *


class MyIntro(Scene):
    # manim -p -ql my_intro.py MyIntro
    def construct(self):
        background = ScreenRectangle(
            height=8,
            stroke_width=0,
            fill_color="#ffffff",
            fill_opacity=1)

        tex = MathTex(
            r"J\upsilon\varsigma\tau G\mu\varsigma",
            font_size=200,
            color=ManimColor.from_hex('#3333B2'),
        )

        g = VGroup(background, tex)

        self.add(background)
        self.play(Write(tex, run_time=2))
        self.wait()
        self.play(FadeOut(g, run_time=2))
        self.wait()
