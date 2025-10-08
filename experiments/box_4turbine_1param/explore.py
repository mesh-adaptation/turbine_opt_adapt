"""Explore the 1D parameter space by sampling a range of control values."""

import argparse
import os

import numpy as np
from firedrake.utility_meshes import RectangleMesh
from goalie.adjoint import AdjointMeshSeq
from goalie.optimisation import OptimisationProgress
from goalie.time_partition import TimeInstant
from setup import OneParameterSetup

from turbine_opt_adapt.qoi import get_qoi
from turbine_opt_adapt.solver import get_solver
from turbine_opt_adapt.test_case_setup import get_initial_condition

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Explore parameter space by varying yc.")
parser.add_argument("--n", type=float, default=0, help="Initial mesh resolution.")
args, _ = parser.parse_known_args()

n = args.n
if np.isclose(n, np.round(n)):
    output_dir = f"outputs/fixed_mesh_{int(n)}"
else:
    output_dir = f"outputs/fixed_mesh_{n:.4f}".replace(".", "p")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Consider a relatively fine uniform mesh
nx = np.round(60 * 2**n).astype(int)
ny = np.round(25 * 2**n).astype(int)
mesh = RectangleMesh(nx, ny, 1200, 500)

# Explore the parameter space and compute the corresponding cost function values
turbine = OneParameterSetup.control2turbine["yc"]
dim = OneParameterSetup.control_dims["yc"]
yl, yu = OneParameterSetup.control_bounds["yc"]
n_sample = int(np.round(yu - yl + 1))
controls = np.linspace((yl + yu) / 2, yu, n_sample)
qois = []
powers = []
bnds = []
gradients = []
for i, control in enumerate(controls):
    print(f"Sample {i + 1} / {n_sample}")

    OneParameterSetup.initial_turbine_coordinates[turbine][dim] = control

    mesh_seq = AdjointMeshSeq(
        TimeInstant(OneParameterSetup.get_fields()),
        mesh,
        get_initial_condition=get_initial_condition,
        get_solver=get_solver,
        get_qoi=get_qoi,
        qoi_type="steady",
        test_case_setup=OneParameterSetup,
    )
    mesh_seq.progress = OptimisationProgress()

    mesh_seq.solve_adjoint(compute_gradient=True)
    J = mesh_seq.J
    print(f"control={control:6.4f}, qoi={J:11.4e}")
    qois.append(J)
    powers.append(mesh_seq.progress["J_power"][-1])
    bnds.append(mesh_seq.progress["J_bnd"][-1])
    gradients.append(float(mesh_seq.gradient["yc"]))

    # Save the trajectory to file
    np.save(f"{output_dir}/sampled_controls.npy", controls[: i + 1])
    np.save(f"{output_dir}/sampled_qois.npy", qois)
    np.save(f"{output_dir}/sampled_powers.npy", powers)
    np.save(f"{output_dir}/sampled_bnds.npy", bnds)
    np.save(f"{output_dir}/sampled_gradients.npy", gradients)
