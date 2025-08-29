"""Plot progress of controls, QoIs, gradients, and DoFs on different axes.

Run `python3 plot_progress.py --help` to see the available command-line options.
"""

import argparse
import os

import matplotlib.pyplot as plt
from setup import SingleParameterSetup

from turbine_opt_adapt.experiment import get_latest_experiment_id
from turbine_opt_adapt.plotting import ProgressPlotter

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
experiment_id = get_latest_experiment_id(git_hash=args.git_hash)
plot_dir = f"plots/{experiment_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for i, label in enumerate(("controls", "qois", "gradients", "dofs")):
    fig, axes = plt.subplots()
    plotter = ProgressPlotter(
        SingleParameterSetup,
        axes,
        "timings",
        label,
        experiment_id,
        args.n,
        n_range,
        targets,
    )
    plotter.plot_all()
    handles, labels = axes.get_legend_handles_labels()
    plt.savefig(f"{plot_dir}/{label}.jpg", bbox_inches="tight")

    if i == 0:
        # Save the legend separately
        fig = plt.figure()
        fig.legend(handles, labels, loc="center", frameon=False, ncol=3)
        fig.savefig(f"{plot_dir}/legend.jpg", bbox_inches="tight")
