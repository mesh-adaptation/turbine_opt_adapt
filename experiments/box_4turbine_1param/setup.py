"""Module containing setup functions for the one-parameter test case."""


from turbine_opt_adapt.test_case_setup import TestCaseSetup

__all__ = ["OneParameterSetup"]


class OneParameterSetup(TestCaseSetup):

    """Class to hold parameters related to the one-parameter test case."""

    initial_turbine_coordinates = [
        [450.0, 250.0],
        [450.0, 310.0],
        [450.0, 190.0],
        [750.0, 260.0],
    ]
    control_turbines = {3: ("yc",)}
    control_dims = {"yc": 1}
    control_bounds = {"yc": (190.0, 310.0)}
    # Rescale the functional such that the gradients are ~ order magnitude 1
    qoi_scaling = 100.0
    initial_velocity = (1e-03, 0.0)
