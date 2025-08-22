"""Module containing plotting routines for tidal energy problems."""

import matplotlib.pyplot as plt
from matplotlib import patches

__all__ = ["add_patch", "plot_patches"]


def add_patch(axes, xloc, yloc, colour, label, diameter=20.0):
    """Add a square patch to the given axes at the specified location.

    :param axes: The matplotlib axes object to which the patch will be added.
    :type axes: matplotlib.axes.Axes
    :param xloc: The x-coordinate of the center of the square.
    :type xloc: float
    :param yloc: The y-coordinate of the center of the square.
    :type yloc: float
    :param colour: The color of the square (used for both edge and face).
    :type colour: str
    :param label: The label associated with the square.
    :type label: str
    :param diameter: The side length of the square. Defaults to 20.0.
    :type diameter: float
    """
    axes.add_patch(
        patches.Rectangle(
            (xloc - diameter / 2, yloc - diameter / 2),
            diameter,
            diameter,
            edgecolor=colour,
            facecolor=colour,
            linewidth=0.1,
            label=label,
        )
    )


def plot_patches(mesh_seq, optimised, filename):
    """Plot the initial and final turbine locations over the mesh.

    :arg mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :arg optimised: dictionary of optimised controls
    :type optimised: dict[float]
    :arg filename: name of the file to save the plot
    :type filename: :class:`str`
    """
    test_case = mesh_seq.test_case_setup
    fig, axes = plt.subplots(figsize=(12, 5))
    mesh_seq.plot(fig=fig, axes=axes)
    for (x, y) in test_case.fixed_turbine_locations:
        add_patch(axes, x, y, "C0", "Fixed turbines")
    for (x, y) in test_case.control_turbine_locations:
        add_patch(axes, x, y, "C1", "Initial control turbines")
    for control, value in optimised.items():
        turbine = test_case.control_turbines[control]
        xy = test_case.turbine_locations[turbine]
        xy[test_case.control_dims[control]] = value
        x, y = xy
        add_patch(axes, x, y, "C2", "Optimised control turbines")
    axes.set_title("")
    handles, labels = axes.get_legend_handles_labels()
    indices = [4, 4 + test_case.num_fixed_turbines, 4 + test_case.num_turbines]
    handles = [handles[i] for i in indices]
    labels = [labels[i] for i in indices]
    axes.legend(handles, labels, loc="upper left")
    plt.savefig(filename, bbox_inches="tight")
