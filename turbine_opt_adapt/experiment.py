"""Module containing functions for setting up experiments."""

import abc
import datetime
import os
import subprocess

import numpy as np

__all__ = [
    "get_latest_experiment_id",
    "FixedMeshExperiment",
    "GoalOrientedExperiment",
]


def get_git_hash(index=None):
    """Get the git hash for a specific commit.

    :param index: The index of the commit to retrieve. If None, retrieves the latest
        commit.
    :return: The short git hash of the commit.
    """
    rev = "HEAD"
    if index is not None:
        if index < 0:
            raise ValueError("Index must be a non-negative integer.")
        elif index > 0:
            rev = f"HEAD~{index}"
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", rev])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as cpe:
        raise RuntimeError("Could not retrieve git hash.") from cpe


def get_experiment_id():
    """Generate experiment identifier with datetime stamp and git hash.

    :return: A string in the format "YYYYMMDD_<git_hash>".
    """
    return f"{datetime.datetime.now().strftime('%Y%m%d')}_{get_git_hash()}"


def get_latest_experiment_id(git_hash=None):
    """Get the latest experiment ID that contains outputs.

    :param git_hash: Optional git hash to filter experiments. If provided, only
        experiments with this git hash will be considered.
    :return: The latest experiment ID in the format "YYYYMMDD_<git_hash>".
    :raises ValueError: If no experiments are found.
    """
    experiments_dir = "outputs"

    if git_hash is not None:
        for experiment_id in os.listdir(experiments_dir):
            if experiment_id.endswith(git_hash):
                return experiment_id
        raise ValueError("No experiments found for the specified git hash.")

    latest = None
    index = 0
    while latest is None:
        try:
            git_hash = get_git_hash(index)
        except subprocess.CalledProcessError as cpe:
            raise ValueError("No experiments found.") from cpe
        for experiment_id in os.listdir(experiments_dir):
            if experiment_id.endswith(git_hash):
                return experiment_id
        index += 1


class Experiment(abc.ABC):
    """A class holding experiment metadata."""

    @abc.abstractmethod
    def __init__(self):
        pass

    def _check_dir_exists(self, name):
        if not hasattr(self, "config_str"):
            raise ValueError("Experiment doesn't have a config_str.")
        dirname = os.path.join(name, self.config_str)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    @property
    def plot_dir(self):
        return self._check_dir_exists("plot")

    @property
    def output_dir(self):
        return self._check_dir_exists("output")

    @property
    def data_dir(self):
        return self._check_dir_exists("data")


class FixedMeshExperiment(Experiment):
    """A class for holding metadata for a fixed mesh experiment."""

    def __init__(self, n):
        """Construct.

        :arg n: the mesh resolution
        :type n: int
        """
        super().__init__()
        if np.isclose(n, np.round(n)):
            self.config_str = f"fixed_mesh_{int(n)}"
        else:
            self.config_str = f"fixed_mesh_{n:.4f}".replace(".", "p")


class GoalOrientedExperiment(Experiment):
    """A class for holding metadata for a goal-oriented experiment."""

    def __init__(self, n, anisotropic, base, target):
        """Construct.

        :arg n: the mesh resolution
        :type n: int
        :arg anisotropic: whether anisotropic adaptation is used
        :type anisotropic: bool
        :arg base: base metric complexity
        :type base: float
        :arg target: target metric complexity
        :type target: float
        """
        super().__init__()
        self.experiment_id = get_experiment_id()
        print(f"Experiment ID: {self.experiment_id}")
        if np.isclose(n, np.round(n)):
            prefix = f"goal_oriented_n{int(n)}"
        else:
            prefix = f"goal_oriented_n{n:.4f}".replace(".", "p")
        self.config_str = "_".join(
            [
                prefix,
                f"anisotropic{int(anisotropic)}",
                f"base{int(base)}",
                f"target{int(target)}",
            ]
        )
