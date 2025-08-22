"""Module containing setup functions for the single-parameter test case."""

import matplotlib.pyplot as plt
import ufl
from firedrake.assemble import assemble
from thetis.utility import domain_constant

from turbine_opt_adapt.plotting import add_patch
from turbine_opt_adapt.test_case_setup import TestCaseSetup

__all__ = ["SingleParameterSetup", "get_qoi", "plot_setup", "plot_patches"]


class SingleParameterSetup(TestCaseSetup):

    """Class to hold parameters related to the single-parameter test case."""

    turbine_locations = [
        [450.0, 250.0],
        [450.0, 310.0],
        [450.0, 190.0],
        [750.0, 260.0],
    ]
    control_indices = {"yc": (3, 1)}
    qoi_scaling = 100.0
    initial_velocity = (1e-03, 0.0)


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


def plot_setup(filename):
    """Plot the initial turbine locations.

    :arg filename: name of the file to save the plot
    :type filename: :class:`str`
    """
    fig, axes = plt.subplots(figsize=(12, 5))
    axes.plot([0, 0], [0, 500], color="C2", linewidth=3, label="Inflow boundary")
    axes.plot([1200, 1200], [0, 500], color="C3", linewidth=3, label="Outflow boundary")
    axes.plot([0, 1200], [0, 0], color="C4", linewidth=3, label="No-slip boundary")
    axes.plot([0, 1200], [500, 500], color="C4", linewidth=3)
    for xloc, yloc in SingleParameterSetup.turbine_locations[:3]:
        add_patch(axes, xloc, yloc, "C0", "Fixed turbines")
    xc, yc = SingleParameterSetup.turbine_locations[3]
    add_patch(axes, xc, yc, "C1", "Control turbine")
    axes.set_title("")
    axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
    axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")
    axes.axis(True)
    eps = 5
    axes.set_xlim([-eps, 1200 + eps])
    axes.set_ylim([-eps, 500 + eps])
    handles, labels = axes.get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[2], handles[-2], handles[-1]]
    labels = [
        "Inflow boundary",
        "Outflow boundary",
        "No-slip boundary",
        labels[-2],
        labels[-1],
    ]
    axes.legend(handles, labels, loc="upper left")
    plt.savefig(filename, bbox_inches="tight")


# TODO: Introduce a function in turbine_opt_adapt that automates this
def plot_patches(mesh_seq, optimised, filename):
    """Plot the initial and final turbine locations over the mesh.

    :arg mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :arg optimised: y-coordinate of the optimised turbine location
    :type optimised: :class:`float`
    :arg filename: name of the file to save the plot
    :type filename: :class:`str`
    """
    fig, axes = plt.subplots(figsize=(12, 5))
    mesh_seq.plot(fig=fig, axes=axes)
    for xloc, yloc in mesh_seq.test_case_setup.turbine_locations[:3]:
        add_patch(axes, xloc, yloc, "C0", "Fixed turbines")
    xc, yc = mesh_seq.test_case_setup.turbine_locations[3]
    add_patch(axes, xc, yc, "C1", "Initial control turbine")
    add_patch(axes, xc, optimised, "C2", "Optimised control turbine")
    axes.set_title("")
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[-3:], labels[-3:], loc="upper left")
    plt.savefig(filename, bbox_inches="tight")
