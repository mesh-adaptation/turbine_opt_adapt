"""Run the two-parameter test case with goal-oriented adaptation.

Run `python3 goal_oriented.py --help` to see the available command-line options.
"""

import argparse
import os
from time import perf_counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from animate.adapt import adapt
from animate.metric import RiemannianMetric
from firedrake.checkpointing import CheckpointFile
from firedrake.functionspace import TensorFunctionSpace
from goalie.log import pyrint
from goalie.metric import ramp_complexity
from goalie.optimisation import QoIOptimiser
from goalie.options import OptimisationParameters
from goalie.plot import plot_indicator_snapshots
from matplotlib import ticker
from setup import TwoParameterSetup

from turbine_opt_adapt.experiment import GoalOrientedExperiment
from turbine_opt_adapt.plotting import plot_patches, plot_solution

# TODO: Avoid duplication with 1-parameter case

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Run with goal-oriented adaptation.")
parser.add_argument("--n", type=float, default=0, help="Initial mesh resolution.")
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
parser.add_argument(
    "--anisotropic",
    action="store_true",
    help="Use anisotropic adaptation (default: False).",
)
parser.add_argument("--plot_fields", action="store_true", help="Plot solution fields.")
args, _ = parser.parse_known_args()
n = args.n
anisotropic = args.anisotropic
base = args.base_complexity
target = args.target_complexity
experiment = GoalOrientedExperiment(n, anisotropic, base, target)

# Set up the GoalOrientedMeshSeq
start_time = perf_counter()
mesh_seq = TwoParameterSetup.create_mesh_seq(n, goal_oriented=True)

if args.plot_fields:
    # Solve the adjoint problem, computing gradients
    solutions = mesh_seq.solve_adjoint(compute_gradient=True)
    J = mesh_seq.J
    print(f"J = {J:.4e}")

    # Plot the x-velocity component of both the forward and adjoint solutions
    plot_solution(solutions, experiment, adjoint=False)
    plot_solution(solutions, experiment, adjoint=True)

# Set optimiser parameters, including a large starting step length
parameters = OptimisationParameters({"lr": 10.0, "gtol": 1.0e-3})
print(parameters)


def adaptor(mesh_seq, solutions, indicators):
    """Adapt the mesh based on the error indicators and Hessian.

    :param mesh_seq: the mesh sequence to adapt.
    :type mesh_seq: :class:`~goalie.go_mesh_seq.GoalOrientedMeshSeq`
    :param solutions: the solutions computed on the mesh sequence.
    :type solutions: :class:`~goalie.function_data.FunctionData`
    :param indicators: the error indicators computed on the mesh sequence.
    :type indicators: :class:`~goalie.function_data.IndicatorData`
    :return: a list of booleans indicating whether to continue adapting the mesh.
    :rtype: list[bool]
    """
    P1_ten = TensorFunctionSpace(mesh_seq[0], "CG", 1)
    metric = RiemannianMetric(P1_ten)

    # Ramp the target metric complexity over the first few iterations
    iteration = mesh_seq.fp_iteration
    mp = {
        "dm_plex_metric_target_complexity": ramp_complexity(
            base,
            target,
            iteration,
            num_iterations=3,
        ),
        "dm_plex_metric_hausdorff_number": 0.01 * 1000,
    }
    metric.set_parameters(mp)

    if anisotropic:
        # Recover the Hessian of the forward solution
        hessians = {key: RiemannianMetric(P1_ten) for key in ("u", "v", "eta")}
        uv, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
        hessians["u"].compute_hessian(uv[0])
        hessians["v"].compute_hessian(uv[1])
        hessians["eta"].compute_hessian(eta)
        # FIXME: Why doesn't intersection work here?
        hessian = hessians["u"].average(hessians["v"], hessians["eta"])

        # Deduce an anisotropic metric from the error indicator field and the Hessian
        metric.compute_anisotropic_dwr_metric(indicators["solution_2d"][0][0], hessian)
        # FIXME: Why does it only refine the inflow?
        # TODO: Plot Hessian components for debugging (indicators seem fine)
    else:
        # Deduce an isotropic metric from the error indicator field
        metric.compute_isotropic_dwr_metric(indicators["solution_2d"][0][0])
    complexity = metric.complexity()

    # Adapt the mesh
    mesh_seq[0] = adapt(mesh_seq[0], metric)
    num_dofs = mesh_seq.count_vertices()[0]
    num_elem = mesh_seq.count_elements()[0]
    pyrint(
        f"{iteration:2d}, complexity: {complexity:4.0f}"
        f", dofs: {num_dofs:4d}, elements: {num_elem:4d}"
    )

    # Write each intermediate adapted mesh
    checkpoint_filename = os.path.join(
        experiment.data_dir, f"{experiment.config_str}_mesh{iteration}.h5"
    )
    with CheckpointFile(checkpoint_filename, "w") as chk:
        chk.save_mesh(mesh_seq[0])

    # Plot error indicator on intermediate meshes
    if args.plot_fields:
        plot_kwargs = {"figsize": (12, 5)}
        plot_kwargs["norm"] = mcolors.LogNorm()
        plot_kwargs["locator"] = ticker.LogLocator()
        fig, axes, tcs = plot_indicator_snapshots(
            indicators, mesh_seq.time_partition, "solution_2d", **plot_kwargs
        )
        axes.set_title(f"Indicator at iteration {iteration}")
        fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
        fig.savefig(
            f"{experiment.plot_dir}/{experiment.config_str}_indicator{iteration}.jpg"
        )
        plt.close()

    # Check whether the target complexity has been (approximately) reached. If not,
    # return ``True`` to indicate that convergence checks should be skipped until the
    # next fixed point iteration.
    continue_unconditionally = complexity < 0.95 * target
    return [continue_unconditionally]


# Run the optimiser
optimiser = QoIOptimiser(
    mesh_seq, ("xc", "yc"), parameters, method="gradient_descent", adaptor=adaptor
)
optimiser.minimise(dropout=False)
cpu_time = perf_counter() - start_time
print(f"Optimisation completed in {cpu_time:.2f} seconds.")

# Write the optimiser progress to file
for key, value in optimiser.progress.items():
    np.save(f"{experiment.output_dir}/{experiment.config_str}_{key}.npy", value)

if args.plot_fields:
    # Plot the patches for the final positions
    optimised = {
        "xc": optimiser.progress["controls"][-1][0],
        "yc": optimiser.progress["controls"][-1][1],
    }
    plot_patches(
        mesh_seq,
        optimised,
        f"{experiment.plot_dir}/{experiment.config_str}_patches.jpg",
    )

    # Plot the x-velocity component of the forward solution for the final control/mesh
    plot_solution(solutions, experiment, optimised=True)
