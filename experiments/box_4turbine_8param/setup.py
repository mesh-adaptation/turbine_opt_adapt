"""Module containing setup functions for the eight-parameter test case."""

from turbine_opt_adapt.test_case_setup import TestCaseSetup

__all__ = ["EightParameterSetup"]


class EightParameterSetup(TestCaseSetup):
    """Class to hold metadata related to the eight-parameter test case."""

    initial_turbine_coordinates = [
        [450.0, 250.0],
        [450.0, 310.0],
        [450.0, 190.0],
        [750.0, 260.0],
    ]
    control_turbines = {
        0: ("x0", "y0"),
        1: ("x1", "y1"),
        2: ("x2", "y2"),
        3: ("x3", "y3"),
    }
    control_dims = {
        "x0": 0,
        "y0": 1,
        "x1": 0,
        "y1": 1,
        "x2": 0,
        "y2": 1,
        "x3": 0,
        "y3": 1,
    }
    control_bounds = {
        "x0": (450.0, 1190.0),
        "y0": (190.0, 310.0),
        "x1": (450.0, 1190.0),
        "y1": (190.0, 310.0),
        "x2": (450.0, 1190.0),
        "y2": (190.0, 310.0),
        "x3": (450.0, 1190.0),
        "y3": (190.0, 310.0),
    }
    # Rescale the functional such that the gradients are ~ order magnitude 1
    qoi_scaling = 100.0
    initial_velocity = (1e-03, 0.0)
