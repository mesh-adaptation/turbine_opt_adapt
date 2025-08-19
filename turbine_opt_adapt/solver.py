"""
Module containing a coastal ocean solver which is compatible with mesh adaptation.
"""
from thetis.solver2d import FlowSolver2d
from thetis.utility import unfrozen


class TurbineSolver2d(FlowSolver2d):
    """
    2D coastal ocean solver which is compatible with steady-state mesh adaptation.

    Subclass of :class:`~thetis.solver2d.FlowSolver2d`.
    """
    @unfrozen
    def __init__(self, mesh_seq, index, bathymetry, options=None):
        """
        Initialise the solver.

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
        """
        Create function spaces for the solver.

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
        """
        Create fields for the solver.

        This method overrides the base class method to ensure the fields are consistent
        with those defined in the mesh sequence.
        """
        super().create_fields()
        self.fields.solution_2d = self.mesh_seq.field_functions["solution_2d"]
        self.fields.uv_2d, self.fields.elev_2d = self.fields.solution_2d.subfunctions
