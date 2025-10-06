"""Explore the 2D parameter space by sampling a range of control values."""

import argparse
import os

import numpy as np
from firedrake.utility_meshes import RectangleMesh
from goalie.adjoint import AdjointMeshSeq
from goalie.optimisation import OptimisationProgress
from goalie.time_partition import TimeInstant
from setup import TwoParameterSetup

from turbine_opt_adapt.qoi import get_qoi
from turbine_opt_adapt.solver import get_solver
from turbine_opt_adapt.test_case_setup import get_initial_condition

# TODO: Avoid duplication with 1D case

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(
    description="Explore parameter space by varying xc and yc."
)
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
turbine = TwoParameterSetup.control2turbine["xc"]
assert turbine == TwoParameterSetup.control2turbine["yc"]
assert TwoParameterSetup.control_dims["xc"] == 0
assert TwoParameterSetup.control_dims["yc"] == 1
xl, xu = TwoParameterSetup.control_bounds["xc"]
yl, yu = TwoParameterSetup.control_bounds["yc"]
n_sample_x = int(np.round(xu - xl + 1))
n_sample_y = int(np.round((yu - yl) / 2 + 1))
x_controls = np.linspace(xl, xu, n_sample_x)
y_controls = np.linspace((yl + yu) / 2, yu, n_sample_y)
n_sample = n_sample_x * n_sample_y
controls = []
qois = []
powers = []
bnds = []
for i, x_control in enumerate(x_controls):
    for j, y_control in enumerate(y_controls):
        k = n_sample_y * i + j + 1
        print(f"Sample {k} / {n_sample}")

        TwoParameterSetup.initial_turbine_coordinates[turbine][0] = x_control
        TwoParameterSetup.initial_turbine_coordinates[turbine][1] = y_control

        mesh_seq = AdjointMeshSeq(
            TimeInstant(TwoParameterSetup.get_fields()),
            mesh,
            get_initial_condition=get_initial_condition,
            get_solver=get_solver,
            get_qoi=get_qoi,
            qoi_type="steady",
            test_case_setup=TwoParameterSetup,
        )
        mesh_seq.progress = OptimisationProgress()

        # FIXME: get_checkpoints gives tiny QoI
        # mesh_seq.get_checkpoints(run_final_subinterval=True)
        mesh_seq.solve_adjoint()
        J = mesh_seq.J
        print(f"x_control={x_control:6.4f}, y_control={y_control:6.4f}, qoi={J:11.4e}")
        controls.append((x_control, y_control))
        qois.append(J)
        powers.append(mesh_seq.progress["J_power"][-1])
        bnds.append(mesh_seq.progress["J_bnd"][-1])

        # Save the trajectory to file
        np.save(f"{output_dir}/sampled_controls.npy", controls[:k])
        np.save(f"{output_dir}/sampled_qois.npy", qois)
        np.save(f"{output_dir}/sampled_powers.npy", powers)
        np.save(f"{output_dir}/sampled_bnds.npy", bnds)
