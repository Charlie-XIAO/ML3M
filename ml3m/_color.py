"""This file is for (possibly) printing colored text to terminal."""


from __future__ import annotations

import sys
from enum import Enum
from typing import Any


class COLOR(Enum):
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    DEFAULT = "\033[39m"


_color_support = True


if sys.platform == "win32":
    try:
        # https://stackoverflow.com/questions/36760127
        from ctypes import windll

        kernel32 = windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:  # pragma: no cover
        _color_support = False


def colored(content: Any, color: COLOR) -> str:
    """Return the content that will be printed colored.

    Parameters
    ----------
    content : str
        The content to color.
    color : COLOR
        The color type.
    """
    if _color_support:
        return f"{color.value}{content}{COLOR.DEFAULT.value}"
    return f"{content}"  # pragma: no cover
