"""Plot values against the 2D parameter space sampled from fixed mesh runs.

Run `python3 plot_progress_parameter_spaces.py --help` to see the available command-line
options.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.pyplot import tricontourf
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

# Load exploration data from file
ny = 61
sampled_controls = np.load(f"{output_dir}/sampled_controls.npy")
int_controls = np.array(np.round(sampled_controls), dtype=np.int32)
assert np.allclose(sampled_controls, int_controls)
control2index = {f"{x}_{y}": i for i, (x, y) in enumerate(int_controls)}
sampled_qois = -np.load(f"{output_dir}/sampled_qois.npy") * scaling
sampled_powers = np.load(f"{output_dir}/sampled_powers.npy") * 1e-6
sampled_bnds = np.load(f"{output_dir}/sampled_bnds.npy") * 1e-6
assert np.allclose(sampled_powers + sampled_bnds, sampled_qois)
nx = sampled_controls.shape[0] // 61

# Avoid infinite values
sampled_qois = np.nan_to_num(sampled_qois, posinf=1e6)
sampled_bnds = np.nan_to_num(sampled_bnds, posinf=1e6)

# Define a mesh based on the sampled controls
xmin, ymin = sampled_controls.min(axis=0)
xmax, ymax = sampled_controls.max(axis=0)
mesh = RectangleMesh(nx-1, ny-1, xmax, ymax, xmin, ymin)
int_coords = np.array(np.round(mesh.coordinates.dat.data), dtype=np.int32)
assert np.allclose(mesh.coordinates.dat.data, int_coords)

# Convert arrays into Firedrake Functions
P1 = FunctionSpace(mesh, "CG", 1)
qois = Function(P1, name="Sampled QoIs")
powers = Function(P1, name="Sampled power outputs")
bnds = Function(P1, name="Sampled log barrier")
for i, (x, y) in enumerate(int_coords):
    idx = control2index[f"{x}_{y}"]
    qois.dat.data[i] = sampled_qois[idx]
    powers.dat.data[i] = sampled_powers[idx]
    bnds.dat.data[i] = sampled_bnds[idx]

# Load the trajectory from the fixed mesh optimisation run
x_traj, y_traj = np.load(f"{output_dir}/{experiment_id}_controls.npy").transpose()
# fixed_mesh_qois = -np.load(f"{output_dir}/{experiment_id}_qois.npy") * scaling

for name, field in zip(("power", "qoi", "bnd"), (powers, qois, bnds)):

    # Plot the parameter space
    fig, axes = plt.subplots(figsize=(12, 5))
    axes.set_title("")
    levels = 9 if name == "bnd" else np.linspace(6, 6.6, 9)
    fig.colorbar(tricontourf(field, axes=axes, cmap="coolwarm", levels=levels), ax=axes)
    axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
    axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")

    # Find the maximum value and annotate the plot
    max_index = field.dat.data.argmax()
    max_control = int_coords[max_index]
    max_power = field.dat.data[max_index]
    print(f"Maximum {name}: {max_power} at Control: {max_control}")
    axes.plot(*max_control, "*", color="k", label="Sampled maximum")

    # Plot the trajectory with the maximum QoI highlighted
    axes.plot(x_traj, y_traj, "--x", color="C0", label="Fixed mesh trajectory")

    # TODO: Add a zoomed-in inset around the maximum value

    axes.legend()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(f"{plot_dir}/progress_{name}_space.png", bbox_inches="tight")
