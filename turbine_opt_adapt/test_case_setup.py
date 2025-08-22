"""Module containing a base class for test case setup."""
import abc

import ufl
from finat.ufl import FiniteElement, MixedElement, VectorElement
from firedrake.function import Function
from goalie.field import Field

__all__ = ["TestCaseSetup", "get_initial_condition"]

class TestCaseSetup(abc.ABC):

    """Base class for holding parameters related to a turbine optimisation test case."""

    turbine_locations = []
    control_turbines = {}
    control_dims = {}
    qoi_scaling = 1.0
    initial_velocity = (0.0, 0.0)

    @property
    def initial_controls(cls):
        """Get the initial control values.

        :return: dictionary of initial control values
        :rtype: dict[float]
        """
        return {
            control: cls.turbine_locations[turbine][cls.control_dims[control]]
            for control, turbine in cls.control_turbines.items()
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
        for control in cls.control_turbines:
            fields.append(
                Field(
                    control,
                    family="Real",
                    degree=0,
                    unsteady=False,
                    solved_for=False,
                )
            )
        return fields

def get_initial_condition(mesh_seq):
    """Get the initial conditions for a turbine optimisation test case.

    :param mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :return: dictionary with initial conditions for the solution and control variable
    :rtype: dict
    """
    solution_2d = Function(mesh_seq.function_spaces["solution_2d"][0])
    u, eta = solution_2d.subfunctions
    u.interpolate(ufl.as_vector(mesh_seq.test_case_setup.initial_velocity))
    eta.assign(0.0)
    ics = {"solution_2d": solution_2d}
    for control, turbine in mesh_seq.test_case_setup.control_turbines.items():
        dim = mesh_seq.test_case_setup.control_dims[control]
        ics[control] = Function(mesh_seq.function_spaces[control][0])
        ics[control].assign(mesh_seq.test_case_setup.turbine_locations[turbine][dim])
    return ics
