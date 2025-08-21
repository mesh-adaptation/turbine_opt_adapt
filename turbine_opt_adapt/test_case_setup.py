"""Module containing a base class for test case setup."""
import abc


class TestCaseSetup(abc.ABC):

    """Base class for holding parameters related to a particular test case."""

    turbine_locations = []
    control_indices = {}
    qoi_scaling = 1.0

    @property
    def initial_controls(cls):
        """Get the initial control values.

        :return: dictionary of initial control values
        :rtype: dict[float]
        """
        return {
            control: cls.turbine_locations[turbine][dim]
            for control, (turbine, dim) in cls.control_indices.items()
        }
