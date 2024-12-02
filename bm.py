from manim import *
import pandas as pd


MY_RED = "#ff7575"
MY_BLUE = "#759cff"

# ===== Functions =====
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


def split(self):
    self.wait(5)
    self.next_section()

Scene.split = split


# ===== Custom Classes =====
class Coin(VMobject):

    def __init__(self, face, radius=0.4, **kwargs):
        super().__init__(**kwargs)

        if face == 'H':
            self.add(
                Circle(color=WHITE, fill_color=BLUE_E,
                       fill_opacity=1, radius=radius, stroke_width=5*radius),
                Tex("H", font_size=135*radius)
            )
        else:
            self.add(
                Circle(color=WHITE, fill_color=RED_E,
                       fill_opacity=1, radius=radius, stroke_width=5*radius),
                Tex("T", font_size=135*radius)
            )

    @override_animation(Create)
    def _create_override(self):
        return AnimationGroup(
            FadeIn(self, scale=1.2, shift=DOWN * 0.2, run_time=0.5),
            self.animate.flip(),
        )


class CoinLine(VMobject):
    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        self.add(*[Coin(t) for t in sequence]
                 ).arrange_in_grid(buff=0.15, cols=10)

    @override_animation(Create)
    def _Create_override(self, lag_ratio=0.5, run_time=1):
        return AnimationGroup(
            *[Create(c) for c in self],
            lag_ratio=lag_ratio,
            run_time=run_time
        )


class CoinSim(VGroup):
    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        wealth = 100
        line = [MathTex(r"\$" + str(wealth))]
        for l in sequence:
            line.append(Coin(l).next_to(line[-1], LEFT))
            factor = 0.5
            if l == 'H':
                factor = 1.8
            line.append(MathTex(r"\times" + str(factor)).next_to(line[-2]))
            wealth *= factor

            line.append(
                Tex(r"\$" + "{:.0f}".format(wealth)).next_to(line[-2]).shift(RIGHT))
        self.line = VGroup(*line)
        self.add(self.line)

    def animate(self, scene):
        scene.play(Write(self.line[0]))
        scene.split()
        i = 0
        for i in range(0, len(self.line)-1, 3):
            scene.play(Create(self.line[i+1]), FadeIn(self.line[i+2]))
            scene.split()
            scene.play(ReplacementTransform(
                VGroup(self.line[i], self.line[i+2]), self.line[i+3]))

        scene.split()
        scene.play(*[FadeOut(self.line[i])
                   for i in range(1, len(self.line), 3)], FadeOut(self.line[-1]))


# ===== Scenes =====
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


class CoinTosses(Scene):
    # manim -p -ql bm.py CoinTosses

    def construct(self):

        n_throws = 8
        w = 0

        plane = NumberPlane(
            x_range=(0, n_throws),
            y_range=(-n_throws, n_throws),
            x_length=11,
            y_length=5,
            axis_config={"include_numbers": True},
        )
        plane.shift(DOWN)
        self.play(Create(plane, run_time=0.5))

        np.random.seed(666)
        for n in range(n_throws):

            # Coin Toss
            randu = np.random.rand()
            if randu <= 0.5:  # Heads
                increment = 1
                face = "H"
                textvalue = r"+1"
                textcolor = BLUE

            else:  # Tails
                increment = -1
                face = "T"
                textvalue = r"-1"
                textcolor = RED

            c = Coin(face=face)
            self.play(Create(c), run_time=0.5)  # Coin appears
            self.play(c.animate.shift(UP * 3 + LEFT * (6.5 - 0.8 * n)), run_time=0.5)  # Move the coin to coin line
            self.play(Write(  # Add the value of the increment
                MathTex(textvalue, color=textcolor)
                .move_to(UP * 2 + LEFT * (6.5 - 0.8 * n)),
                run_time=0.5,
            ))

            # Increment the line chart
            line_graph = plane.plot_line_graph(
                x_values=[n, n + 1],
                y_values=[w, w + increment],
                line_color=GOLD_E,
                vertex_dot_style=dict(stroke_width=3, fill_color=GOLD_E),
                stroke_width=4,
            )
            self.play(Create(line_graph, run_time=0.5))

            # Set up next iteration
            w = w + increment


class Fractal(Scene):
    # manim -p -ql bm.py Fractal
    def construct(self):

        n = 15  # number of morphs into fractal
        t = 2 ** n  # Number of observations
        df = simulate_srw(t=t, n=1, k=1)
        df = df['SRW 1']

        axes = Axes(
            x_range=(df.index[0], df.index[-1], 10),
            y_range=(df.min(), df.max(), 5),
            axis_config={"include_ticks": False, "include_numbers": False},
            tips=False,
        )
        labels = axes.get_axis_labels(
            Tex(r"$t$").scale(1), Tex(r"$dW_t$").scale(1)
        )

        self.play(
            Write(axes, run_time=1),
            Write(labels, run_time=1),
        )

        # Draw the first line
        index2plot = np.linspace(start=0, stop=t, num=2)
        series2plot = df.loc[index2plot].copy()
        coords = [axes.c2p(x, y) for x, y in zip(series2plot.index, series2plot.values)]
        plot = VMobject(color=BLUE).set_points_as_corners(coords).set_stroke(width=1)
        self.play(Write(plot, run_time=1))
        self.wait(0.2)

        for n in range(1, n + 1):
            index2plot = np.linspace(start=0, stop=t, num=2**n + 1)
            series2plot = df.loc[index2plot]
            coords = [axes.c2p(x, y) for x, y in zip(series2plot.index, series2plot.values)]
            new_plot = VMobject(color=BLUE).set_points_as_corners(coords).set_stroke(width=1)
            self.play(Transform(plot, new_plot, run_time=1))
            self.wait(0.2)


CoinTosses().construct()