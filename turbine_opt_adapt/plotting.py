"""Module containing plotting routines for tidal energy problems."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

__all__ = ["plot_box_setup", "plot_patches", "ProgressPlotter"]


def plot_box_setup(filename, test_case):
    """Plot the initial turbine locations for problems in box domains.

    :arg filename: name of the file to save the plot
    :type filename: :class:`str`
    :arg test_case: class defining test case
    :type test_case: :class:`~turbine_opt_adapt.test_case_setup.TestCaseSetup`
    """
    fig, axes = plt.subplots(figsize=(12, 5))
    axes.plot([0, 0], [0, 500], color="C2", linewidth=3, label="Inflow boundary")
    axes.plot([1200, 1200], [0, 500], color="C3", linewidth=3, label="Outflow boundary")
    axes.plot([0, 1200], [0, 0], color="C4", linewidth=3, label="No-slip boundary")
    axes.plot([0, 1200], [500, 500], color="C4", linewidth=3)
    for (x, y) in test_case.fixed_turbine_coordinates:
        add_patch(axes, x, y, "C0", "Fixed turbines")
    for (x, y) in test_case.initial_control_turbine_coordinates:
        add_patch(axes, x, y, "C1", "Control turbine")
    axes.set_title("")
    axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
    axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")
    axes.axis(True)
    eps = 5
    axes.set_xlim([-eps, 1200 + eps])
    axes.set_ylim([-eps, 500 + eps])
    handles, labels = axes.get_legend_handles_labels()
    indices = [0, 1, 2, 4, 4 + test_case.num_fixed_turbines]
    handles = [handles[i] for i in indices]
    labels = [labels[i] for i in indices]
    axes.legend(handles, labels, loc="upper left")
    plt.savefig(filename, bbox_inches="tight")


def add_patch(axes, xloc, yloc, colour, label, diameter=20.0):
    """Add a square patch to the given axes at the specified location.

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


def plot_patches(mesh_seq, optimised, filename):
    """Plot the initial and final turbine locations over the mesh.

    :arg mesh_seq: mesh sequence holding the mesh
    :type mesh_seq: :class:`goalie.mesh_seq.MeshSeq`
    :arg optimised: dictionary of optimised controls
    :type optimised: dict[float]
    :arg filename: name of the file to save the plot
    :type filename: :class:`str`
    """
    test_case = mesh_seq.test_case_setup
    fig, axes = plt.subplots(figsize=(12, 5))
    mesh_seq.plot(fig=fig, axes=axes)
    for (x, y) in test_case.fixed_turbine_coordinates:
        add_patch(axes, x, y, "C0", "Fixed turbines")
    for (x, y) in test_case.initial_control_turbine_coordinates:
        add_patch(axes, x, y, "C1", "Initial control turbines")
    for control, value in optimised.items():
        turbine = test_case.control_turbines[control]
        xy = test_case.initial_turbine_coordinates[turbine]
        xy[test_case.control_dims[control]] = value
        x, y = xy
        add_patch(axes, x, y, "C2", "Optimised control turbines")
    axes.set_title("")
    handles, labels = axes.get_legend_handles_labels()
    indices = [4, 4 + test_case.num_fixed_turbines, 4 + test_case.num_turbines]
    handles = [handles[i] for i in indices]
    labels = [labels[i] for i in indices]
    axes.legend(handles, labels, loc="upper left")
    plt.savefig(filename, bbox_inches="tight")


class ProgressPlotter:

    """Class to encapsulate the plotting logic for progress of controls and QoIs."""

    base = 1000  # Base value for target complexity

    def __init__(
        self,
        test_case_setup,
        axes,
        x,
        y,
        experiment_id,
        base_n,
        n_range,
        targets,
    ):
        """Initialise the plotter.

        :param test_case_setup: class containing test case configuration.
        :param axes: Matplotlib axes to plot on.
        :param x: Name of the x-axis variable (e.g., "timings").
        :param y: Name of the y-axis variable (e.g., "controls").
        :param experiment_id: Identifier for the experiment.
        :param base_n: Base mesh resolution for goal-oriented approaches.
        :param n_range: Range of mesh resolutions to consider.
        :param targets: List of target complexities for goal-oriented approaches.
        """
        self.test_case_setup = test_case_setup
        self.axes = axes
        self.x = x
        self.y = y
        self.experiment_id = experiment_id
        self.plot_dir = f"plots/{experiment_id}"
        self.base_n = base_n
        self.n_range = n_range
        self.targets = targets

    @property
    def scaling(self):
        """Scaling for QoIs and gradients."""
        return -1e-6 / self.test_case_setup.qoi_scaling

    @property
    def labels(self):
        """Dictionary of variables to plot and their labels."""
        labels = {
            "controls": r"Control turbine position [$\mathrm{m}$]",
            "qois": r"Power output [$\mathrm{MW}$]",
            "dofs": "DoF count",
        }
        if self.test_case_setup.num_control_turbines == 1:
            labels["gradients"] = "Gradient relative to initial value"
        return labels

    def try_load(self, run, variable, output_dir):
        """Attempt to load a numpy array for a given run and variable name.

        :param run: Identifier for the run (e.g., "fixed_mesh_1").
        :param variable: Name of the variable to load (e.g., "timings").
        :param output_dir: Directory where the output data is stored.
        :return: Loaded numpy array or None if not found.
        """
        fname = f"{output_dir}/{run}_{variable}.npy"
        try:
            arr = np.load(fname)
        except FileNotFoundError as file_error:
            print(f"File '{fname}' not found.")
            raise FileNotFoundError from file_error
        if variable in ("qois", "gradients"):
            arr *= self.scaling
        if variable == "gradients":
            arr = np.abs(arr)
            arr /= arr[0]  # Normalise by the first value
        return arr

    def try_plot(self, run, label, output_dir):
        """Attempt to plot data for a given run and variable names on the provided axes.

        :param run: Identifier for the run (e.g., "fixed_mesh_1").
        :param label: Label for the plot line.
        :param output_dir: Directory where the output data is stored.
        """
        try:
            x_arr = self.try_load(run, self.x, output_dir)
            y_arr = self.try_load(run, self.y, output_dir)
        except FileNotFoundError:
            return
        if self.y == "gradients":
            self.axes.loglog(x_arr, y_arr, "--x", label=label)
        else:
            self.axes.semilogx(x_arr, y_arr, "--x", label=label)

    @staticmethod
    def n_str(n):
        """Convert a mesh resolution `n` to a string representation."""
        return n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")

    def plot_fixed_mesh(self, n):
        """Plot data for a fixed mesh with a given resolution.

        :param n: Mesh resolution.
        """
        run = f"fixed_mesh_{self.n_str(n)}"
        output_dir = f"outputs/{run}"
        try:
            dofs = self.try_load(run, "dofs", output_dir)[-1]
        except FileNotFoundError:
            return
        label = f"Fixed mesh ({dofs:.0f} DoFs)"
        self.try_plot(run, label, output_dir)

    def plot_goal_oriented(self, n, anisotropic, target):
        """Plot data for a goal-oriented approach.

        :param n: Mesh resolution.
        :param anisotropic: Bool indicating if the approach is anisotropic.
        :param target: Target complexity for the goal-oriented approach.
        """
        aniso = int(anisotropic)
        go_name = ["Isotropic goal-oriented", "Anisotropic goal-oriented"][aniso]
        label = rf"{go_name} ($\mathcal{{C}}_T={target:.0f}$)"
        run = "_".join(
            [
                "goal_oriented",
                f"n{self.n_str(n)}",
                f"anisotropic{aniso}",
                f"base{self.base:.0f}",
                f"target{target:.0f}",
            ]
        )
        self.try_plot(run, label, output_dir=f"outputs/{self.experiment_id}")

    def plot_all(self):
        """Plot data across all configurations."""
        for n in self.n_range:
            self.plot_fixed_mesh(n)
        for anisotropic in range(2):
            for target in self.targets:
                self.plot_goal_oriented(self.base_n, anisotropic, target)
        self.axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
        self.axes.set_ylabel(self.labels[self.y])
        self.axes.grid(True)
