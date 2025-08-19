"""
Explore the parameter space by sampling a range of control values.
"""
import argparse

from firedrake import *
from firedrake.pyplot import *
from goalie import *
from setup import *

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Explore parameter space by varying yc.")
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
args, _ = parser.parse_known_args()

n = args.n
output_dir = f"outputs/fixed_mesh_{n}"

# Consider a relatively fine uniform mesh
mesh = RectangleMesh(60 * 2**n, 25 * 2**n, 1200, 500)

# Explore the parameter space and compute the corresponding cost function values
y1, y2 = turbine_locations[0][1], turbine_locations[1][1]
controls = np.linspace(y1, y2, int(np.round(2 * (y2 - y1) + 1)))
qois = []
for i, control in enumerate(controls):
    def get_ic(*args):
        return get_initial_condition(*args, init_control=control)

    mesh_seq = AdjointMeshSeq(
        TimeInstant(fields),
        mesh,
        get_initial_condition=get_ic,
        get_solver=get_solver,
        get_qoi=get_qoi,
        qoi_type="steady",
    )

    # FIXME: get_checkpoints gives tiny QoI
    # mesh_seq.get_checkpoints(run_final_subinterval=True)
    mesh_seq.solve_adjoint()
    J = mesh_seq.J
    print(f"control={control:6.4f}, qoi={J:11.4e}")
    qois.append(J)

    # Save the trajectory to file
    np.save(f"{output_dir}/sampled_controls.npy", controls[: i + 1])
    np.save(f"{output_dir}/sampled_qois.npy", qois)
