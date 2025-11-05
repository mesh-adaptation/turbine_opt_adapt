"""Run the eight-parameter test case on a fixed mesh.

Run `python3 fixed_mesh.py --help` to see the available command-line options.
"""

import argparse
import os
from time import perf_counter

import numpy as np
from firedrake.utility_meshes import RectangleMesh
from goalie.adjoint import AdjointMeshSeq
from goalie.optimisation import QoIOptimiser
from goalie.options import OptimisationParameters
from goalie.time_partition import TimeInstant
from setup import EightParameterSetup

from turbine_opt_adapt.plotting import plot_box_setup, plot_patches, plot_solution
from turbine_opt_adapt.qoi import get_qoi
from turbine_opt_adapt.solver import get_solver
from turbine_opt_adapt.test_case_setup import get_initial_condition

# TODO: Avoid duplication with 1- and 2-parameter cases

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Run the optimiser on a fixed mesh.")
parser.add_argument("--n", type=float, default=0, help="Initial mesh resolution.")
parser.add_argument("--taylor_test", action="store_true", help="Run a Taylor test.")
parser.add_argument("--plot_setup", action="store_true", help="Plot the problem setup.")
parser.add_argument("--plot_fields", action="store_true", help="Plot solution fields.")
args, _ = parser.parse_known_args()

# Use parsed arguments
n = args.n

if np.isclose(n, np.round(n)):
    experiment_id = f"fixed_mesh_{int(n)}"
else:
    experiment_id = f"fixed_mesh_{n:.4f}".replace(".", "p")
plot_dir = f"plots/{experiment_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
output_dir = f"outputs/{experiment_id}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_time = perf_counter()

# Set up the AdjointMeshSeq
nx = np.round(60 * 2**n).astype(int)
ny = np.round(25 * 2**n).astype(int)
mesh_seq = AdjointMeshSeq(
    TimeInstant(EightParameterSetup.get_fields()),
    RectangleMesh(nx, ny, 1200, 500),
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
    test_case_setup=EightParameterSetup,
)

# Plot the problem setup
if args.plot_setup:
    plot_box_setup("plots/setup.jpg", EightParameterSetup)

# Run a Taylor test to check the gradient is computed correctly
if args.taylor_test:
    mesh_seq.taylor_test("yc")

if args.plot_fields:
    # Solve the adjoint problem, computing gradients
    solutions = mesh_seq.solve_adjoint(compute_gradient=True)
    J = mesh_seq.J
    print(f"J = {J:.4e}")

    # Plot the x-velocity component of both the forward and adjoint solutions
    for mode in ("forward", "adjoint"):
        plot_solution(
            solutions["solution_2d"][mode],
            f"{plot_dir}/{experiment_id}_{mode}_unoptimised.jpg",
        )

# Set optimiser parameters, including a large starting step length
parameters = OptimisationParameters({"lr": 10.0, "gtol": 1.0e-3})
print(parameters)

# Run the optimiser
controls = ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3")
optimiser = QoIOptimiser(mesh_seq, controls, parameters, method="gradient_descent")
solutions = optimiser.minimise()

cpu_time = perf_counter() - start_time
print(f"Optimisation completed in {cpu_time:.2f} seconds.")
with open(f"{output_dir}/cputime.txt", "w") as f:
    f.write(f"{cpu_time:.2f} seconds\n")

# Write the optimiser progress to file
for key, value in optimiser.progress.items():
    np.save(f"{output_dir}/{experiment_id}_{key}.npy", value)

if args.plot_fields:
    # Plot the patches for the final positions in relatively low resolution cases
    if n < 2:
        optimised = {
            "xc": optimiser.progress["controls"][-1][0],
            "yc": optimiser.progress["controls"][-1][1],
        }
        plot_patches(mesh_seq, optimised, f"{plot_dir}/patches.jpg")

    # Plot the x-velocity component of the forward solution for the final control
    plot_solution(
        solutions["solution_2d"]["forward"],
        f"{plot_dir}/{experiment_id}_forward_optimised.jpg",
    )
