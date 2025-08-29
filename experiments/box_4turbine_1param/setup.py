"""Module containing setup functions for the single-parameter test case."""

import ufl
from firedrake.assemble import assemble
from thetis.utility import domain_constant

from turbine_opt_adapt.test_case_setup import TestCaseSetup

__all__ = ["SingleParameterSetup", "get_qoi"]


class SingleParameterSetup(TestCaseSetup):

    """Class to hold parameters related to the single-parameter test case."""

    turbine_locations = [
        [450.0, 250.0],
        [450.0, 310.0],
        [450.0, 190.0],
        [750.0, 260.0],
    ]
    control_turbines = {"yc": 3}
    control_dims = {"yc": 1}
    qoi_scaling = 100.0
    initial_velocity = (1e-03, 0.0)


# TODO: Separate regularisation into test case setup; generalise the rest
def get_qoi(mesh_seq, index):
    """Get the quantity of interest (QoI) functional for the single-parameter test case.

    :param mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :param index: index of the mesh in the sequence
    :type index: :class:`int`
    :return: function that computes the QoI
    :rtype: function
    """

    def steady_qoi():
        mesh = mesh_seq[index]
        u, eta = ufl.split(mesh_seq.field_functions["solution_2d"])
        yc = mesh_seq.field_functions["yc"]
        farm = mesh_seq.tidal_farm
        farm_options = mesh_seq.tidal_farm_options

        # Power output contribution
        J_power = farm.turbine.power(u, eta) * farm.turbine_density * ufl.dx

        # Add a regularisation term for constraining the control
        area = assemble(domain_constant(1.0, mesh) * ufl.dx)
        alpha = domain_constant(1.0 / area, mesh)
        y2 = farm_options.turbine_coordinates[1][1]
        y3 = farm_options.turbine_coordinates[2][1]
        J_reg = (
            alpha
            * ufl.conditional(
                yc < y3, (yc - y3) ** 2, ufl.conditional(yc > y2, (yc - y2) ** 2, 0)
            )
            * ufl.dx
        )

        # Sum the two components
        # NOTE: We rescale the functional such that the gradients are ~ order magnitude
        #       1
        # NOTE: We also multiply by -1 so that if we minimise the functional, we
        #       maximise power (maximize is also available from pyadjoint but currently
        #       broken)
        J_overall = mesh_seq.test_case_setup.qoi_scaling * (-J_power + J_reg)
        # print(
        #     f"DEBUG: power={assemble(J_power)}, reg={assemble(J_reg)},"
        #     f" overall={assemble(J_overall)}"
        # )
        return J_overall

    return steady_qoi
