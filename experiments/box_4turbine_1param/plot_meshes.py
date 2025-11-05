"""Plot meshes from a given goal-oriented simulation.

Run `python3 plot_meshes.py --help` to see the available command-line options.
"""

# TODO: Generalise for other experiments

import argparse
import os

import matplotlib.pyplot as plt
from firedrake.checkpointing import CheckpointFile
from firedrake.pyplot import triplot

from turbine_opt_adapt.experiment import GoalOrientedExperiment

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Plot meshes from a given simulation.")
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument(
    "--anisotropic",
    action="store_true",
    help="Use anisotropic adaptation (default: False).",
)
parser.add_argument(
    "--base_complexity",
    type=float,
    default=1000.0,
    help="Base metric complexity (default: 1000.0).",
)
parser.add_argument(
    "--target_complexity",
    type=float,
    default=1000.0,  # FIXME: Avoid adjoint solver fail with larger values
    help="Target metric complexity (default: 1000.0).",
)
args = parser.parse_args()
n = args.n
anisotropic = args.anisotropic
base = args.base_complexity
target = args.target_complexity
experiment = GoalOrientedExperiment(n, anisotropic, base, target)
config_str = experiment.config_str
plot_dir = experiment.plot_dir
data_dir = experiment.data_dir

# Write each intermediate adapted mesh
file_exists = True
iteration = 1
while file_exists:
    checkpoint_filename = os.path.join(data_dir, f"{config_str}_mesh{iteration}.h5")
    file_exists = os.path.exists(checkpoint_filename)
    if not file_exists:
        break

    with CheckpointFile(checkpoint_filename, "w") as chk:
        # FIXME: Topology name is wrong
        mesh = chk.load_mesh()

    fig, axes = plt.subplots(figsize=(12, 5))
    interior_kw = {"edgecolor": "k", "linewidth": 0.5}
    triplot(mesh, axes=axes, interior_kw=interior_kw)
    axes.set_title(f"Mesh at iteration {iteration}")
    fig.savefig(f"{plot_dir}/{config_str}_mesh{iteration}.jpg")
    plt.close()

    iteration += 1
