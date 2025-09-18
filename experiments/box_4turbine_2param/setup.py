"""Module containing setup functions for the two-parameter test case."""

import ufl
from firedrake.assemble import assemble
from thetis.utility import domain_constant

from turbine_opt_adapt.test_case_setup import TestCaseSetup

__all__ = ["TwoParameterSetup"]


class TwoParameterSetup(TestCaseSetup):

    """Class to hold parameters related to the two-parameter test case."""

    initial_turbine_coordinates = [
        [450.0, 250.0],
        [450.0, 310.0],
        [450.0, 190.0],
        [750.0, 260.0],
    ]
    control_turbines = {3: ("xc", "yc")}
    control_dims = {"xc": 0, "yc": 1}
    # Rescale the functional such that the gradients are ~ order magnitude 1
    qoi_scaling = 100.0
    initial_velocity = (1e-03, 0.0)

    @classmethod
    def regularisation_term(cls, mesh_seq, index):
        """Add a regularisation term for constraining the control.

        :param mesh_seq: mesh sequence holding the mesh
        :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
        :param index: index of the mesh in the sequence
        :type index: :class:`int`
        :return: regularisation term
        :rtype: :class:`~.ufl.Expr`
        """
        mesh = mesh_seq[index]
        xc = mesh_seq.field_functions["xc"]
        yc = mesh_seq.field_functions["yc"]
        farm_options = mesh_seq.tidal_farm_options
        area = assemble(domain_constant(1.0, mesh) * ufl.dx)
        alpha2 = domain_constant(1.0 / area ** 2, mesh)
        x1 = farm_options.turbine_coordinates[0][0]
        y2 = farm_options.turbine_coordinates[1][1]
        y3 = farm_options.turbine_coordinates[2][1]
        xmax = domain_constant(1200, mesh)
        return (
            alpha2
            * ufl.conditional(
                xc < x1, (xc - x1) ** 2, ufl.conditional(xc > xmax, (xc - xmax) ** 2, 0)
            )
            * ufl.conditional(
                yc < y3, (yc - y3) ** 2, ufl.conditional(yc > y2, (yc - y2) ** 2, 0)
            )
            * ufl.dx
        )
