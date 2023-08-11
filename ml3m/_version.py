"""This file is for version-related functionalities."""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from ._color import COLOR, colored


def show_versions():
    """Print useful debugging information."""
    from . import __version__

    # Adapted from the scikit-learn implementation
    print()
    welcome_msg = f"Welcome to ml3m {__version__}"
    print(welcome_msg)
    print("=" * len(welcome_msg))

    # Print system related information
    print()
    print(colored("System Information", COLOR.GREEN))
    pyver, (pybuildno, pybuilddt) = platform.python_version(), platform.python_build()
    print(f"Python       {pyver} ({pybuildno}, {pybuilddt})")
    print(f"Compiler     {platform.python_compiler()}")
    print(f"Executable   {sys.executable}")
    print(f"Machine      {platform.platform()}")

    # Print python dependencies
    print()
    print(colored("Python dependencies", COLOR.GREEN))
    packages = ["pip", "setuptools", "numpy", "openai", "pandas", "tqdm"]
    for package in packages:
        try:
            package_ver = version(package)
        except PackageNotFoundError:
            package_ver = None
        print(f"{package:<13}{package_ver}")
