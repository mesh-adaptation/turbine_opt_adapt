"""Module containing setup functions for the two-parameter test case."""


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
    # TODO: Properly impose that turbines can't come too close to each other
    control_bounds = {"xc": (450.0, 1190.0), "yc": (190.0, 310.0)}
    # Rescale the functional such that the gradients are ~ order magnitude 1
    qoi_scaling = 100.0
    initial_velocity = (1e-03, 0.0)
