"""
Module containing functions for setting up experiments.
"""
import datetime
import os
import subprocess

__all__ = ["get_experiment_id", "get_latest_experiment_id"]


def get_git_hash(index=None):
    """
    Get the git hash for a specific commit.

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
    """
    Generate experiment identifier with datetime stamp and git hash.

    :return: A string in the format "YYYYMMDD_<git_hash>".
    """
    return f"{datetime.datetime.now().strftime("%Y%m%d")}_{get_git_hash()}"


def get_latest_experiment_id(git_hash=None):
    """
    Get the latest experiment ID that contains outputs.

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
