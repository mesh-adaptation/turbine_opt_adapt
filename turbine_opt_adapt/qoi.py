"""Module containing a QoI getter based on power output."""

import ufl
from firedrake.assemble import assemble

__all__ = ["get_qoi"]

def get_qoi(mesh_seq, index):
    """Get the quantity of interest (QoI) functional for the single-parameter test case.

    :param mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :param index: index of the mesh in the sequence
    :type index: :class:`int`
    :return: function that computes the QoI
    :rtype: function
    """
    if hasattr(mesh_seq, "progress") and ("J_power" not in mesh_seq.progress):
        mesh_seq.progress["J_power"] = []
        mesh_seq.progress["J_bnd"] = []
        mesh_seq.progress["J_reg"] = []

    def steady_qoi():
        u, eta = ufl.split(mesh_seq.field_functions["solution_2d"])
        farm = mesh_seq.tidal_farm

        # Power output contribution
        # NOTE: Negative so that minimising the functional maximises power (maximize is
        #       also available from pyadjoint but currently broken)
        J_power = -farm.turbine.power(u, eta) * farm.turbine_density * ufl.dx
        if hasattr(mesh_seq, "progress"):
            mesh_seq.progress["J_power"].append(assemble(J_power))

        # Log-barrier contribution for imposing bounds
        J_bnd = mesh_seq.test_case_setup.bound_term(mesh_seq, index)
        if hasattr(mesh_seq, "progress"):
            mesh_seq.progress["J_bnd"].append(assemble(J_bnd))

        # Tikhonov regularisation contribution
        J_reg = mesh_seq.test_case_setup.regularisation_term(mesh_seq, index)
        if hasattr(mesh_seq, "progress"):
            mesh_seq.progress["J_reg"].append(assemble(J_reg))

        return mesh_seq.test_case_setup.qoi_scaling * (J_power + J_reg + J_bnd)

    return steady_qoi
