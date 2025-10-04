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
    axes.set_title("")

    # Plot domain boundary
    axes.plot([0, 0], [0, 500], color="C2", linewidth=3, label="Inflow boundary")
    axes.plot([1200, 1200], [0, 500], color="C3", linewidth=3, label="Outflow boundary")
    axes.plot([0, 1200], [0, 0], color="C4", linewidth=3, label="No-slip boundary")
    axes.plot([0, 1200], [500, 500], color="C4", linewidth=3)
    axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
    axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")

    # Add feasible region in the case of a single control turbine
    if test_case.num_controls == 1:
        control = tuple(test_case.control_bounds.keys())[0]
        bounds = test_case.control_bounds[control]
        turbine = test_case.control2turbine[control]
        x, y = test_case.initial_turbine_coordinates[turbine]
        if test_case.control_dims[control] == 0:
            axes.plot(bounds, [y, y], "C1", linewidth=2, label="Feasible region")
        else:
            axes.plot([x, x], bounds, "C1", linewidth=2, label="Feasible region")
    elif test_case.num_controls == 2 and test_case.num_control_turbines == 1:
        dim2control = {dim: control for control, dim in test_case.control_dims.items()}
        axes.fill_between(
            test_case.control_bounds[dim2control[0]],
            *test_case.control_bounds[dim2control[1]],
            color="C1",
            alpha=0.3,
            label="Feasible region"
        )

    # Add patches for the fixed turbines
    for i, (x, y) in enumerate(test_case.fixed_turbine_coordinates):
        add_patch(axes, x, y, "C0", label="Fixed turbines" if i == 0 else None)

    # Add patches for the control turbines
    for i, (x, y) in enumerate(test_case.initial_control_turbine_coordinates):
        add_patch(axes, x, y, "C1", label="Control turbine" if i == 0 else None)

    axes.axis(True)
    eps = 5
    axes.set_xlim([-eps, 1200 + eps])
    axes.set_ylim([-eps, 500 + eps])
    axes.legend(loc="upper left")
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
    axes.set_title("")

    # Plot the domain
    mesh_seq.plot(fig=fig, axes=axes)

    # Add patches for the fixed turbines
    for i, (x, y) in enumerate(test_case.fixed_turbine_coordinates):
        add_patch(axes, x, y, "C0", label="Fixed turbines" if i == 0 else None)

    # Add patches for the initial positions of the control turbines
    for i, (x, y) in enumerate(test_case.initial_control_turbine_coordinates):
        label = "Initial control turbines" if i == 0 else None
        add_patch(axes, x, y, "C1", label=label)

    # Add patches for the optimised positions of the control turbines
    for turbine, controls in test_case.control_turbines.items():
        xy = test_case.initial_turbine_coordinates[turbine]
        for control in controls:
            dim = test_case.control_dims[control]
            xy[dim] = optimised[control]
        x, y = xy
        add_patch(axes, x, y, "C2", label="Optimised control turbines")

    axes.legend(loc="upper left")
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
    def labels(self):
        """Dictionary of variables to plot and their labels."""
        labels = {
            "controls": r"Control turbine position [$\mathrm{m}$]",
            "dofs": "DoF count",
            "qois": r"Power output [$\mathrm{MW}$]",
        }
        if self.test_case_setup.num_controls == 1:
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
            arr *= -1e-6 / self.test_case_setup.qoi_scaling
        elif variable in ("J_power", "J_bnd"):
            arr *= 1e-6
        if variable == "gradients":
            arr = np.abs(arr)
            arr /= arr[0]  # Normalise by the first value
        return arr

    def try_plot(self, run, label, colour, output_dir):
        """Attempt to plot data for a given run and variable names on the provided axes.

        :param run: Identifier for the run (e.g., "fixed_mesh_1").
        :param label: Label for the plot line.
        :param colour: Colour to use for plotting
        :param output_dir: Directory where the output data is stored.
        """
        try:
            x_arr = self.try_load(run, self.x, output_dir)
            y_arr = self.try_load(run, self.y, output_dir)
        except FileNotFoundError:
            return
        if self.y == "gradients":
            self.axes.loglog(x_arr, y_arr, "--x", color=colour, label=label)
        else:
            self.axes.semilogx(x_arr, y_arr, "--x", color=colour, label=label)
        if self.y == "qois":
            try:
                p_arr = self.try_load(run, "J_power", output_dir)
                b_arr = self.try_load(run, "J_bnd", output_dir)
            except FileNotFoundError:
                return
            assert np.allclose(p_arr + b_arr, y_arr)
            self.axes.semilogx(x_arr, p_arr, ":^", color=colour)


    @staticmethod
    def n_str(n):
        """Convert a mesh resolution `n` to a string representation."""
        return n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")

    def plot_fixed_mesh(self, n, colour):
        """Plot data for a fixed mesh with a given resolution.

        :param n: Mesh resolution.
        :param colour: Colour to use for plotting
        """
        run = f"fixed_mesh_{self.n_str(n)}"
        output_dir = f"outputs/{run}"
        try:
            dofs = self.try_load(run, "dofs", output_dir)[-1]
        except FileNotFoundError:
            return
        label = f"Fixed mesh ({dofs:.0f} DoFs)"
        self.try_plot(run, label, colour, output_dir)

    def plot_goal_oriented(self, n, anisotropic, target, colour):
        """Plot data for a goal-oriented approach.

        :param n: Mesh resolution.
        :param anisotropic: Bool indicating if the approach is anisotropic.
        :param target: Target complexity for the goal-oriented approach.
        :param colour: Colour to use for plotting
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
        self.try_plot(run, colour, label, output_dir=f"outputs/{self.experiment_id}")

    def plot_all(self):
        """Plot data across all configurations."""
        for count, n in enumerate(self.n_range):
            self.plot_fixed_mesh(n, f"C{count}")
        for anisotropic in range(2):
            for target in self.targets:
                count += 1
                self.plot_goal_oriented(self.base_n, anisotropic, target, f"C{count}")
        self.axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
        self.axes.set_ylabel(self.labels[self.y])
        self.axes.grid(True)
