"""Module containing a base class for test case setup."""
import abc

import ufl
from finat.ufl import FiniteElement, MixedElement, VectorElement
from goalie.field import Field


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

    @classmethod
    def get_fields(cls):
        """Get the fields for the test case.

        :return: list of fields
        :rtype: list[Field]
        """
        p1dg_element = FiniteElement(
            "Discontinuous Lagrange", ufl.triangle, 1, variant="equispaced"
        )
        p1dgv_element = VectorElement(p1dg_element, dim=2)
        p1dgvp1dg_element = MixedElement([p1dgv_element, p1dg_element])
        fields = [
            Field("solution_2d", finite_element=p1dgvp1dg_element, unsteady=False)
        ]
        for control in cls.control_indices:
            fields.append(
                Field(control,family="Real", degree=0, unsteady=False, solved_for=False)
            )
        return fields
