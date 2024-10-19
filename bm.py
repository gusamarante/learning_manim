from manim import *
import pandas as pd


def simulate_srw(t, n, k=1, random_seed=None):
    """
    Simulates the scaled symmetric random walk that leads to the brownian motion
    :param t: number of years (or whole periods) in the simulation
    :param n: number of steps per period t in the simulated trajectories
    :param k: number of trajectories to simulate
    :param random_seed: random seed for numpy's RNG
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    time_index = np.arange(n * t + 1) / n

    # Scaled Random Walk
    omega = np.random.uniform(size=(t * n, k))  # "flip the coins"
    X = (omega >= 0.5) * 1 + (omega < 0.5) * (-1)  # get the increments
    M = X.cumsum(axis=0)  # Sum the increments (integration)
    M = (1 / np.sqrt(n)) * M  # Scale the process
    M = np.vstack([np.zeros((1, k)), M])  # add a zero as a starting point

    column_names = [f'SRW {i+1}' for i in range(k)]
    simulated_trajectories = pd.DataFrame(
        index=time_index,
        data=M,
        columns=column_names,
    )
    return simulated_trajectories


class Intro(Scene):
    # manim -p -ql bm.py Intro
    def construct(self):

        # ===== Physics =====
        this_color = random_bright_color()
        title_physics = Text(
            "Physics",
            color=this_color,
        )
        tex_physics = MathTex(
            r"\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}",
            color=this_color,
        )
        g_physics = VGroup(title_physics, tex_physics).arrange(DOWN)
        self.play(Write(g_physics))
        self.wait()

        # ===== Biology =====
        this_color = random_bright_color()
        title_bio = Text(
            "Biology",
            color=this_color,
        )
        tex_bio = MathTex(
            r"\frac{dN}{dt} &= \alpha N - \beta N P \\ "
            r"\frac{dP}{dt} &= \delta N P - \gamma P",
            color=this_color,
        )
        g_bio = VGroup(title_bio, tex_bio).arrange(DOWN)
        self.play(Transform(g_physics, g_bio, replace_mobject_with_target_in_scene=True))
        self.wait()

        # ===== Economics =====
        this_color = random_bright_color()
        title_econ = Text(
            "Economics",
            color=this_color,
        )
        tex_econ = MathTex(
            r"\frac{dk}{dt} = s \cdot f(k) - (\delta + n) \cdot k",
            color=this_color,
        )
        g_econ = VGroup(title_econ, tex_econ).arrange(DOWN)
        self.play(Transform(g_bio, g_econ, replace_mobject_with_target_in_scene=True))
        self.wait()

        # ===== Finance =====
        this_color = random_bright_color()
        title_fin = Text(
            "Finance",
            color=this_color,
        )
        tex_fin = MathTex(
            r"\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0",
            color=this_color,
        )
        g_fin = VGroup(title_fin, tex_fin).arrange(DOWN)
        self.play(Transform(g_econ, g_fin, replace_mobject_with_target_in_scene=True))
        self.wait()

        # ===== Fade Out =====
        self.play(FadeOut(g_fin))


class SimulatedTrajectories(Scene):
    # manim -p -ql bm.py SimulatedTrajectories
    def construct(self):

        t = 200
        n = 1
        k = 20

        df = simulate_srw(t=t, n=n, k=k)

        axes = Axes(
            x_range=(df.index[0], df.index[-1], 10),
            y_range=(df.min().min(), df.max().max(), 5),
            axis_config={"include_ticks": False, "include_numbers": False},
            tips=False,
        )
        labels = axes.get_axis_labels(
            Tex("Throws").scale(0.45), Text("Cumulative Sum").scale(0.45)
        )

        self.play(
            Write(axes, run_time=2),
            Write(labels, run_time=2),
        )

        for ii in range(k):
            coords = [axes.c2p(x, y) for x, y in zip(df.index, df[f'SRW {ii+1}'].values)]
            plot = VMobject(color=random_bright_color()).set_points_as_corners(coords)

            self.play(Write(plot, run_time=1))


class Title(Scene):
    # manim -p -ql bm.py Title
    def construct(self):
        tex = MathTex(r"W_t", font_size=144)
        title = Text("Brownian Motion", font_size=100, color=BLUE)
        g = VGroup(tex, title).arrange(DOWN)
        self.play(Write(g))
