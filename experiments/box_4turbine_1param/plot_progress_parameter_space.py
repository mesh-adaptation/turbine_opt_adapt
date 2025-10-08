"""Plot values against the 1D parameter space sampled from fixed mesh runs.

Run `python3 plot_progress_parameter_spaces.py --help` to see the available command-line
options.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CubicSpline
from setup import OneParameterSetup

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
scaling = 1e-6 / OneParameterSetup.qoi_scaling

sampled_controls = np.load(f"{output_dir}/sampled_controls.npy")
sampled_qois = -np.load(f"{output_dir}/sampled_qois.npy") * scaling
sampled_powers = np.load(f"{output_dir}/sampled_powers.npy") * 1e-6
sampled_bnds = np.load(f"{output_dir}/sampled_bnds.npy") * 1e-6
assert np.allclose(-(-sampled_powers + sampled_bnds), sampled_qois)

# Perform cubic interpolation away from boundary
cubic_spline = CubicSpline(sampled_controls[10:-10], sampled_qois[10:-10])

# Find the derivative of the cubic spline
derivative = cubic_spline.derivative()

# Find the critical points (where the derivative is zero)
critical_controls = derivative.roots()

# Evaluate the cubic spline at the critical points
critical_qois = cubic_spline(critical_controls)

# Find the maximum QoI and its corresponding control
max_index = np.argmax(critical_qois)
max_control = critical_controls[max_index]
max_qoi = critical_qois[max_index]
print(f"Maximum QoI: {max_qoi} at Control: {max_control}")

fixed_mesh_controls = np.load(f"{output_dir}/{experiment_id}_controls.npy")
fixed_mesh_qois = -np.load(f"{output_dir}/{experiment_id}_qois.npy") * scaling

# Plot the trajectory with the maximum QoI highlighted
fig, axes = plt.subplots()
axes.plot(sampled_controls, sampled_qois, "--x", color="C0", label="Sampled QoI")
axes.plot(sampled_controls, sampled_powers, ":+", color="C0", label="Sampled power")
axes.plot(max_control, max_qoi, "o", color="C1", label="Maximum value")
axes.plot(fixed_mesh_controls, fixed_mesh_qois, "--^", color="C2", label="Fixed mesh")
axes.plot(
    fixed_mesh_controls[-1],
    fixed_mesh_qois[-1],
    "*",
    color="C2",
    label="Converged value"
)
axes.set_xlabel(r"Control turbine position [$\mathrm{m}$]")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
axes.legend(loc="upper right")

# Add a zoomed-in inset around the maximum value
ax_inset = inset_axes(axes, width="40%", height="20%", loc="center right")
ax_inset.plot(sampled_controls, sampled_qois, "--x")
ax_inset.plot(max_control, max_qoi, "o")
ax_inset.plot(fixed_mesh_controls[-1], fixed_mesh_qois[-1], "*")
ax_inset.set_xlim(max_control - 2.0, max_control + 2.0)
ax_inset.set_ylim(max_qoi - 0.002, max_qoi + 0.002)
ax_inset.grid(True)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plt.savefig(f"{plot_dir}/progress_parameter_space.png", bbox_inches="tight")
