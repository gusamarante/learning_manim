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


class Title(Scene):
    # manim -p -ql bm.py Title
    def construct(self):
        tex = MathTex(r"W_t", font_size=144)
        title = Text("Brownian Motion", font_size=100, color=BLUE)
        g = VGroup(tex, title).arrange(DOWN)
        self.play(Write(g))


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
