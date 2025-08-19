"""Plot progress of controls, QoIs, gradients, and DoFs on different axes.

Run `python3 plot_progress.py --help` to see the available command-line options.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from setup import qoi_scaling

from turbine_opt_adapt.experiment import get_latest_experiment_id


class ProgressPlotter:

    """Class to encapsulate the plotting logic for progress of controls and QoIs."""

    labels = {
        "controls": r"Control turbine position [$\mathrm{m}$]",
        "qois": r"Power output [$\mathrm{MW}$]",
        "gradients": "Gradient relative to initial value",
        "dofs": "DoF count",
    }
    base = 1000  # Base value for target complexity
    scaling = -1e-6 / qoi_scaling # Scaling factor for QoIs and gradients

    def __init__(self, axes, x, y, experiment_id, base_n, n_range, targets):
        """Initialise the plotter.

        :param axes: Matplotlib axes to plot on.
        :param x: Name of the x-axis variable (e.g., "timings").
        :param y: Name of the y-axis variable (e.g., "controls").
        :param experiment_id: Identifier for the experiment.
        :param base_n: Base mesh resolution for goal-oriented approaches.
        :param n_range: Range of mesh resolutions to consider.
        :param targets: List of target complexities for goal-oriented approaches.
        """
        self.axes = axes
        self.x = x
        self.y = y
        self.experiment_id = experiment_id
        self.plot_dir = f"plots/{experiment_id}"
        self.base_n = base_n
        self.n_range = n_range
        self.targets = targets

    def try_load(self, run, variable, output_dir):
        """Attempt to load a numpy array for a given run and variable name.

        :param run: Identifier for the run (e.g., "fixed_mesh_1").
        :param variable: Name of the variable to load (e.g., "timings").
        :param output_dir: Directory where the output data is stored.
        :return: Loaded numpy array or None if not found.
        """
        fname = f"{output_dir}/{run}_{variable}.npy"
        try:
            arr = np.load(fname)
        except FileNotFoundError as file_error:
            print(f"File '{fname}' not found.")
            raise FileNotFoundError from file_error
        if variable in ("qois", "gradients"):
            arr *= self.scaling
        if variable == "gradients":
            arr = np.abs(arr)
            arr /= arr[0]  # Normalise by the first value
        return arr

    def try_plot(self, run, label, output_dir):
        """Attempt to plot data for a given run and variable names on the provided axes.

        :param run: Identifier for the run (e.g., "fixed_mesh_1").
        :param label: Label for the plot line.
        :param output_dir: Directory where the output data is stored.
        """
        try:
            x_arr = self.try_load(run, self.x, output_dir)
            y_arr = self.try_load(run, self.y, output_dir)
        except FileNotFoundError:
            return
        if self.y == "gradients":
            self.axes.loglog(x_arr, y_arr, "--x", label=label)
        else:
            self.axes.semilogx(x_arr, y_arr, "--x", label=label)

    @staticmethod
    def n_str(n):
        """Convert a mesh resolution `n` to a string representation."""
        return n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")

    def plot_fixed_mesh(self, n):
        """Plot data for a fixed mesh with a given resolution.

        :param n: Mesh resolution.
        """
        run = f"fixed_mesh_{self.n_str(n)}"
        output_dir = f"outputs/{run}"
        try:
            dofs = self.try_load(run, "dofs", output_dir)[-1]
        except FileNotFoundError:
            return
        label = f"Fixed mesh ({dofs:.0f} DoFs)"
        self.try_plot(run, label, output_dir)

    def plot_goal_oriented(self, n, anisotropic, target):
        """Plot data for a goal-oriented approach.

        :param n: Mesh resolution.
        :param anisotropic: Bool indicating if the approach is anisotropic.
        :param target: Target complexity for the goal-oriented approach.
        """
        aniso = int(anisotropic)
        go_name = ["Isotropic goal-oriented", "Anisotropic goal-oriented"][aniso]
        label = rf"{go_name} ($\mathcal{{C}}_T={target:.0f}$)"
        run = "_".join([
            "goal_oriented",
            f"n{self.n_str(n)}",
            f"anisotropic{aniso}",
            f"base{self.base:.0f}",
            f"target{target:.0f}",
        ])
        self.try_plot(run, label, output_dir=f"outputs/{self.experiment_id}")

    def plot_all(self):
        """Plot data across all configurations."""
        for n in self.n_range:
            self.plot_fixed_mesh(n)
        for anisotropic in range(2):
            for target in self.targets:
                self.plot_goal_oriented(self.base_n, anisotropic, target)
        axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
        axes.set_ylabel(self.labels[self.y])
        axes.grid(True)

parser = argparse.ArgumentParser(
    description="Plot progress of controls and QoIs on different axes."
)
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument(
    "--git_hash", type=str, default=None, help="Git hash identifier for the experiment."
)
args = parser.parse_args()
targets = [1000, 2000, 4000]
n_range = [0, 1, 2, 2.5850]
exp_id = get_latest_experiment_id(git_hash=args.git_hash)
plot_dir = f"plots/{exp_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for label in ProgressPlotter.labels:
    fig, axes = plt.subplots()
    plotter = ProgressPlotter(axes, "timings", label, exp_id, args.n, n_range, targets)
    plotter.plot_all()
    legend_handles, _ = axes.get_legend_handles_labels()
    plt.savefig(f"{plot_dir}/{label}.jpg", bbox_inches="tight")

# Create a separate figure for the legend
legend_fig = plt.figure()
legend_fig.legend(
    legend_handles,
    ProgressPlotter.labels.values(),
    loc="center",
    frameon=False,
    ncol=len(ProgressPlotter.labels)
)
legend_fig.savefig(f"{plot_dir}/legend.jpg", bbox_inches="tight")
