"""
Module containing plotting routines for tidal energy problems.
"""
import matplotlib.patches as patches

__all__ = ["add_patch"]


def add_patch(axes, xloc, yloc, colour, label, diameter=20.0):
    """
    Adds a square patch to the given axes at the specified location.

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
