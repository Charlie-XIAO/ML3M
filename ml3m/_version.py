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
    print(f"Welcome to ml3m {__version__}")

    # Print system related information
    print()
    print(colored("System Information", COLOR.GREEN))
    print(f"Python         {platform.python_version()} {platform.python_build()}")
    print(f"Compiler       {platform.python_compiler()}")
    print(f"Executable     {sys.executable}")
    print(f"Machine        {platform.platform()}")

    # Print python dependencies
    print()
    print(colored("Python dependencies", COLOR.GREEN))
    for package in ["pip", "setuptools", "numpy", "openai", "pandas", "tqdm"]:
        try:
            package_ver = version(package)
        except PackageNotFoundError:
            package_ver = None
        print(f"{package:<15}{package_ver}")
