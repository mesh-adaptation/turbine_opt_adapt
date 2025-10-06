"""Plot values against the 2D parameter space sampled from fixed mesh runs.

Run `python3 plot_progress_parameter_spaces.py --help` to see the available command-line
options.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from firedrake.pyplot import triplot
from firedrake.utility_meshes import RectangleMesh

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from setup import TwoParameterSetup

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(
    description="Plot progress of controls and QoIs on the same axis."
)
parser.add_argument("--n", type=float, default=0, help="Initial mesh resolution.")
args = parser.parse_args()

n = args.n
if np.isclose(n, np.round(n)):
    experiment_id = f"fixed_mesh_{int(n)}"
else:
    experiment_id = f"fixed_mesh_{n:.4f}".replace(".", "p")
output_dir = f"outputs/{experiment_id}"
plot_dir = f"plots/{experiment_id}"
scaling = 1e-6 / TwoParameterSetup.qoi_scaling

ny = 61
sampled_controls = np.load(f"{output_dir}/sampled_controls.npy").reshape(-1, ny, 2)
sampled_qois = -np.load(f"{output_dir}/sampled_qois.npy").reshape(-1, ny) * scaling
sampled_powers = np.load(f"{output_dir}/sampled_powers.npy").reshape(-1, ny) * 1e-6
sampled_bnds = np.load(f"{output_dir}/sampled_bnds.npy").reshape(-1, ny) * 1e-6
assert np.allclose(sampled_powers + sampled_bnds, sampled_qois)
nx = sampled_controls.shape[0]

xmin = sampled_controls[:,:,0].min()
xmax = sampled_controls[:,:,0].max()
ymin = sampled_controls[:,:,1].min()
ymax = sampled_controls[:,:,1].max()

mesh = RectangleMesh(nx, ny, xmax-xmin, ymax-ymin, xmin, ymin)

# TODO: Convert arrays into Firedrake Functions and plot

fig, axes = plt.subplots()
triplot(mesh, axes=axes)

# TODO: Find the maximum QoI and its corresponding control by querying Functions
# max_control =
# max_qoi =
# print(f"Maximum QoI: {max_qoi} at Control: {max_control}")

fixed_mesh_controls = np.load(f"{output_dir}/{experiment_id}_controls.npy")
fixed_mesh_qois = -np.load(f"{output_dir}/{experiment_id}_qois.npy") * scaling

# TODO: Plot the trajectory with the maximum QoI highlighted

# TODO: Add a zoomed-in inset around the maximum value

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plt.savefig(f"{plot_dir}/progress_parameter_space.png", bbox_inches="tight")
