"""Module containing a base class for test case setup."""
import abc

import ufl
from finat.ufl import FiniteElement, MixedElement, VectorElement
from firedrake.function import Function
from goalie.field import Field
from thetis.utility import domain_constant

__all__ = ["TestCaseSetup", "get_initial_condition"]

class TestCaseSetup(abc.ABC):

    """Base class for holding parameters related to a turbine optimisation test case."""

    initial_turbine_coordinates = []  # list of 2D coordinates
    control_turbines = {}  # key: turbine, value: tuple of variable names
    control_dims = {}  # key: variable name, value: dimension 0 or 1
    control_bounds = {}  # key: variable name, value: 2-tuple with lower and upper bound
    qoi_scaling = 1.0
    initial_velocity = (0.0, 0.0)
    regularisation_coefficient = 0  # TODO: Turn on regularisation
    log_barrier_coefficient = 1000.0  # NOTE: Gets multiplied by qoi_scaling

    @classmethod
    @property
    def control2turbine(cls):
        """Map from control variable to turbine index.

        :return: turbine index
        :rtype: int
        """
        return {
            control: turbine
            for turbine, controls in cls.control_turbines.items()
            for control in controls
        }

    @classmethod
    @property
    def num_turbines(cls):
        """Get the number of turbines.

        :return: number of turbines
        :rtype: int
        """
        return len(cls.initial_turbine_coordinates)

    @classmethod
    @property
    def initial_control_turbine_coordinates(cls):
        """Get the locations of the control turbines.

        :return: list of control locations
        :rtype: list[float]
        """
        return [
            cls.initial_turbine_coordinates[turbine]
            for turbine in cls.control_turbines
        ]

    @classmethod
    @property
    def num_control_turbines(cls):
        """Get the number of control turbines.

        :return: number of control turbines
        :rtype: int
        """
        return len(cls.control_turbines)

    @classmethod
    @property
    def fixed_turbine_coordinates(cls):
        """Get the locations of the fixed turbines.

        :return: list of fixed turbine locations
        :rtype: list[tuple[float, float]]
        """
        return [
            location
            for turbine, location in enumerate(cls.initial_turbine_coordinates)
            if turbine not in cls.control_turbines
        ]

    @classmethod
    @property
    def num_fixed_turbines(cls):
        """Get the number of fixed turbines.

        :return: number of fixed turbines
        :rtype: int
        """
        return cls.num_turbines - cls.num_control_turbines

    @classmethod
    @property
    def num_controls(cls):
        """Get the number of control variables.

        :return: number of control variables
        :rtype: int
        """
        return sum([len(controls) for controls in cls.control_turbines.values()])

    @classmethod
    @property
    def initial_controls(cls):
        """Get the initial control values.

        :return: dictionary of initial control values
        :rtype: dict[float]
        """
        return {
            control: cls.initial_turbine_coordinates[turbine][cls.control_dims[control]]
            for turbine, controls in cls.control_turbines.items()
            for control in controls
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
        for controls in cls.control_turbines.values():
            for control in controls:
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

    @classmethod
    def bound_term(cls, mesh_seq, index):
        """Add a log-barrier term to impose bounds on the control variables.

        :param mesh_seq: mesh sequence holding the mesh
        :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
        :param index: index of the mesh in the sequence
        :type index: :class:`int`
        :return: boundary term
        :rtype: :class:`~.ufl.Expr`
        """
        mesh = mesh_seq.meshes[index]
        tau = domain_constant(cls.log_barrier_coefficient, mesh)
        summation = 0
        for control in cls.control_dims:
            lower, upper = cls.control_bounds[control]
            x = mesh_seq.field_functions[control]
            summation += -ufl.ln(x - lower) - ufl.ln(upper - x)
        return (1.0 / tau) * summation * ufl.dx

    @classmethod
    def regularisation_term(cls, mesh_seq, index):
        """Add a Tikhonov regularisation term.

        :param mesh_seq: mesh sequence holding the mesh
        :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
        :param index: index of the mesh in the sequence
        :type index: :class:`int`
        :return: regularisation term
        :rtype: :class:`~.ufl.Expr`
        """
        mesh = mesh_seq.meshes[index]
        alpha = domain_constant(cls.regularisation_coefficient, mesh)
        summation = sum(
            mesh_seq.field_functions[control] ** 2 for control in cls.control_dims
        )
        return alpha * summation * ufl.dx

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
    for turbine, controls in mesh_seq.test_case_setup.control_turbines.items():
        for control in controls:
            dim = mesh_seq.test_case_setup.control_dims[control]
            ics[control] = Function(mesh_seq.function_spaces[control][0])
            ics[control].assign(mesh_seq.test_case_setup.initial_turbine_coordinates[turbine][dim])
    return ics
