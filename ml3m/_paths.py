"""This file is for path-related functionalities."""


import os
from pathlib import Path


def ensure_path(path: str | Path, is_directory=False) -> str:
    """Create medium directories ensure path is safe for creation.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to ensure creation safety of.
    is_directory : bool, default=False
        Whether ``path`` is a directory instead of a file.

    Returns
    -------
    directories : str
        The path to the directory of ``path`` if ``path`` is a file. Otherwise this is
        ``path`` itself. In both cases, this would be a formatted absolute path.
    """
    directories = path if is_directory else os.path.split(path)[0]
    if not os.path.exists(directories):
        os.makedirs(directories)
    return os.path.abspath(directories)


def validate_path(path: str | Path) -> None:
    """Validate a path.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to validate.
    """
    if not os.path.exists(path):
        raise ValueError(f"Path '{os.path.abspath(path)}' does not exist.")
