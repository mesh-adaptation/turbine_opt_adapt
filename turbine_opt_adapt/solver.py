"""Module containing a coastal ocean solver compatible with mesh adaptation."""
import ufl
from firedrake.constant import Constant
from firedrake.function import Function
from goalie.go_mesh_seq import GoalOrientedMeshSeq
from thetis.options import DiscreteTidalTurbineFarmOptions
from thetis.solver2d import FlowSolver2d
from thetis.utility import domain_constant, get_functionspace, unfrozen

__all__ = ["get_solver"]


class TurbineSolver2d(FlowSolver2d):

    """2D coastal ocean solver which is compatible with steady-state mesh adaptation.

    Subclass of :class:`~thetis.solver2d.FlowSolver2d`.
    """

    @unfrozen
    def __init__(self, mesh_seq, index, bathymetry, options=None):
        """Initialise the solver.

        :param mesh_seq: mesh sequence to use for the solver
        :type mesh_seq: :class:`~goalie.mesh_seq.MeshSeq`
        :param index: index of the mesh in the sequence to use for the solver
        :type index: int
        :param bathymetry: bathymetry function on the mesh
        :type bathymetry: :class:`~firedrake.function.Function`
        :param options: solver options, defaults to None
        :type options: :class:`~thetis.options.Options`, optional
        """
        super().__init__(mesh_seq[index], bathymetry, options=options)
        self.mesh_seq = mesh_seq
        self.index = index

    def create_function_spaces(self):
        """Create function spaces for the solver.

        This method overrides the base class method to ensure the function spaces are
        consistent with those defined in the mesh sequence.
        """
        super().create_function_spaces()
        mesh_seq = self.mesh_seq
        self.function_spaces.V_2d = mesh_seq.function_spaces["solution_2d"][self.index]
        self.function_spaces.U_2d, self.function_spaces.H_2d = (
            self.function_spaces.V_2d.subspaces
        )

    def create_fields(self):
        """Create fields for the solver.

        This method overrides the base class method to ensure the fields are consistent
        with those defined in the mesh sequence.
        """
        super().create_fields()
        self.fields.solution_2d = self.mesh_seq.field_functions["solution_2d"]
        self.fields.uv_2d, self.fields.elev_2d = self.fields.solution_2d.subfunctions


def get_solver(mesh_seq):
    """Get the solver function for the single-parameter test case.

    :param mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :return: solver function that can be used to solve the problem
    :rtype: function
    """

    def solver(index):
        mesh = mesh_seq[index]
        u, eta = mesh_seq.field_functions["solution_2d"].subfunctions

        # Specify bathymetry
        x, y = ufl.SpatialCoordinate(mesh)
        channel_depth = domain_constant(40.0, mesh)
        channel_width = domain_constant(500.0, mesh)
        bathymetry_scaling = domain_constant(2.0, mesh)
        P1_2d = get_functionspace(mesh, "CG", 1)
        y_prime = y - channel_width / 2
        bathymetry = Function(P1_2d)
        bathymetry.interpolate(
            channel_depth - (bathymetry_scaling * y_prime / channel_width) ** 2
        )

        # Setup solver
        solver_obj = TurbineSolver2d(mesh_seq, index, Constant(channel_depth))
        options = solver_obj.options
        options.element_family = "dg-dg"
        options.timestep = 1.0
        options.simulation_export_time = 1.0
        options.simulation_end_time = 0.5
        options.no_exports = True
        options.swe_timestepper_type = "SteadyState"
        options.swe_timestepper_options.solver_parameters = {
            "snes_rtol": 1.0e-12,
        }
        options.swe_timestepper_options.ad_block_tag = "solution_2d"
        # options.use_grad_div_viscosity_term = False
        options.horizontal_viscosity = Constant(0.5)
        options.quadratic_drag_coefficient = Constant(0.0025)
        # options.use_grad_depth_viscosity_term = False

        # Setup boundary conditions
        solver_obj.bnd_functions["shallow_water"] = {
            1: {"uv": Constant((3.0, 0.0))},
            2: {"elev": Constant(0.0)},
            3: {"un": Constant(0.0)},
            4: {"un": Constant(0.0)},
        }
        solver_obj.create_function_spaces()

        # Define the thrust curve of the turbine using a tabulated approach:
        # speeds_AR2000: speeds for corresponding thrust coefficients - thrusts_AR2000
        # thrusts_AR2000: list of idealised thrust coefficients of an AR2000 tidal
        # turbine using a curve fitting technique with:
        #   * cut-in speed = 1 m/s
        #   * rated speed = 3.05 m/s
        #   * cut-out speed = 5 m/s
        # (ramp up and down to cut-in and at cut-out speeds for model stability)
        # NOTE: Taken from Thetis:
        #    https://github.com/thetisproject/thetis/blob/master/examples/discrete_turbines/tidal_array.py
        speeds_AR2000 = [
            0.0,
            0.75,
            0.85,
            0.95,
            1.0,
            3.05,
            3.3,
            3.55,
            3.8,
            4.05,
            4.3,
            4.55,
            4.8,
            5.0,
            5.001,
            5.05,
            5.25,
            5.5,
            5.75,
            6.0,
            6.25,
            6.5,
            6.75,
            7.0,
        ]
        thrusts_AR2000 = [
            0.010531,
            0.032281,
            0.038951,
            0.119951,
            0.516484,
            0.516484,
            0.387856,
            0.302601,
            0.242037,
            0.197252,
            0.16319,
            0.136716,
            0.115775,
            0.102048,
            0.060513,
            0.005112,
            0.00151,
            0.00089,
            0.000653,
            0.000524,
            0.000442,
            0.000384,
            0.000341,
            0.000308,
        ]

        # Setup tidal farm
        farm_options = DiscreteTidalTurbineFarmOptions()
        turbine_density = Function(solver_obj.function_spaces.P1_2d).assign(1.0)
        farm_options.turbine_type = "table"
        farm_options.turbine_density = turbine_density
        farm_options.turbine_options.diameter = 20.0
        farm_options.turbine_options.thrust_speeds = speeds_AR2000
        farm_options.turbine_options.thrust_coefficients = thrusts_AR2000
        farm_options.upwind_correction = False
        farm_options.turbine_coordinates = [
            [domain_constant(xloc, mesh), domain_constant(yloc, mesh)]
            for (xloc, yloc) in mesh_seq.test_case_setup.initial_turbine_coordinates
        ]
        for control, turbine in mesh_seq.test_case_setup.control_turbines.items():
            dim = mesh_seq.test_case_setup.control_dims[control]
            farm_options.turbine_coordinates[turbine][dim] = (
                mesh_seq.field_functions[control]
            )
        options.discrete_tidal_turbine_farms["everywhere"] = [farm_options]

        # Apply initial conditions and solve
        solver_obj.assign_initial_conditions(uv=u, elev=eta)
        solver_obj.iterate()

        # Communicate variational form to mesh_seq
        if isinstance(mesh_seq, GoalOrientedMeshSeq):
            mesh_seq.read_forms({"solution_2d": solver_obj.timestepper.F})

        # Stash info related to tidal farms
        mesh_seq.tidal_farm_options = farm_options
        assert len(solver_obj.tidal_farms) == 1
        mesh_seq.tidal_farm = solver_obj.tidal_farms[0]

        yield

    return solver
