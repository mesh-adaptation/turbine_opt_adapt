import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from setup import qoi_scaling
from utils import get_latest_experiment_id

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(
    description="Plot progress of controls and QoIs on different axes."
)
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument(
    "--hash", type=str, default=None, help="Git hash identifier for the experiment."
)
# TODO: Accept multiple target complexities
args = parser.parse_args()
base = 1000
targets = [1000] # TODO: Vary complexity
n_range = [0, 1, 2, 2.5850]
scaling = 1e-6 / qoi_scaling
experiment_id = get_latest_experiment_id(hash=args.hash)
plot_dir = f"plots/{experiment_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Define approaches
labels = {
    "controls": r"Control turbine position [$\mathrm{m}$]",
    "qois": r"Power output [$\mathrm{MW}$]",
    "gradients": "Gradient relative to initial value",
    "dofs": "DoF count",
}

def try_load(run, variable, output_dir):
    """
    Attempt to load a numpy array for a given run and variable name.

    :param run: Identifier for the run (e.g., "fixed_mesh_1").
    :param variable: Name of the variable to load (e.g., "timings").
    :param output_dir: Directory where the output data is stored (default: "outputs").
    :return: Loaded numpy array or None if not found.
    """
    fname = f"{output_dir}/{run}_{variable}.npy"
    try:
        return np.load(fname)
    except FileNotFoundError as file_error:
        print(f"File '{fname}' not found.")
        raise FileNotFoundError from file_error

def try_plot(axes, x, y, run, label, output_dir):
    """
    Attempt to plot data for a given run and variable names on the provided axes.

    :param axes: Matplotlib axes to plot on.
    :param x: Name of the x-axis variable (e.g., "timings").
    :param y: Name of the y-axis variable (e.g., "controls").
    :param run: Identifier for the run (e.g., "fixed_mesh_1").
    :param label: Label for the plot line.
    :param output_dir: Directory where the output data is stored (default: "outputs").
    """
    try:
        x_arr = try_load(run, x, output_dir)
        y_arr = try_load(run, x, output_dir)
    except FileNotFoundError:
        return
    if y == "gradients":
        y_arr = np.abs(y_arr) * scaling
        y_arr /= y_arr[0]  # Normalise by the first value
    axes.semilogx(x_arr, y_arr, "--x", label=label)

def plot_fixed_mesh(axes, n, x, y):
    """
    Plot data for a fixed mesh with a given resolution `n` on the provided axes.

    :param axes: Matplotlib axes to plot on.
    :param n: Mesh resolution.
    :param x: Name of the x-axis variable (e.g., "timings").
    :param y: Name of the y-axis variable (e.g., "controls").
    """
    n_str = n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")
    run = f"fixed_mesh_{n_str}"
    output_dir = f"outputs/{run}"
    try:
        dofs = try_load(run, x, output_dir)[-1]
    except FileNotFoundError:
        return
    label = f"Fixed mesh ({dofs:.0f} DoFs)"
    try_plot(axes, x, y, run, label, output_dir)

def plot_goal_oriented(axes, n, anisotropic, target, x, y):
    """
    Plot data for a goal-oriented approach with a given mesh resolution `n`,
    anisotropy, and target complexity on the provided axes.

    :param axes: Matplotlib axes to plot on.
    :param n: Mesh resolution.
    :param anisotropic: Bool indicating if the approach is anisotropic.
    :param target: Target complexity for the goal-oriented approach.
    :param x: Name of the x-axis variable (e.g., "timings").
    :param y: Name of the y-axis variable (e.g., "controls").
    """
    n_str = n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")
    aniso = int(anisotropic)
    go_name = ["Isotropic goal-oriented", "Anisotropic goal-oriented"][aniso]
    label = rf"{go_name} ($\mathcal{{C}}_T={target:.0f}$)"
    run = f"goal_oriented_n{n}_anisotropic{aniso}_base{base:.0f}_target{target:.0f}"
    try_plot(axes, x, y, run, label, output_dir=f"outputs/{experiment_id}")

def plot_all(axes, x, y):
    """
    Plot all results on the provided axes, for all mesh resolutions and goal-oriented
    approaches requested.

    :param axes: Matplotlib axes to plot on.
    :param x: Name of the x-axis variable (e.g., "timings").
    :param y: Name of the y-axis variable (e.g., "controls").
    """
    for n in n_range:
        plot_fixed_mesh(axes, n, x, y)
    for anisotropic in range(2):
        for target in targets:
            plot_goal_oriented(axes, args.n, anisotropic, target, x, y)
    axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
    axes.set_ylabel(labels[y])
    axes.grid(True)
    axes.legend()

# Plot results for all dependent variables
for label in labels:
    fig, axes = plt.subplots()
    plot_all(axes, "timings", label)
    plt.savefig(f"{plot_dir}/{label}.jpg", bbox_inches="tight")
