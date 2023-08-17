from __future__ import annotations

import shutil
import sys
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ._typing import DataItemType

#######################################################################################
#                                                                                     #
#                                        COLOR                                        #
#                                                                                     #
#######################################################################################


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


def colored(content: Any, color: COLOR | None) -> str:
    """Return the content that will be printed colored.

    Parameters
    ----------
    content : str
        The content to color.
    color : COLOR
        The color type.

    Returns
    -------
    colored_content : str
        The colored content. Default color if coloring is not supported.
    """
    if _color_support and color is not None:
        return f"{color.value}{content}{COLOR.DEFAULT.value}"
    return f"{content}"


#######################################################################################
#                                                                                     #
#                                        EMOJI                                        #
#                                                                                     #
#######################################################################################


class EMOJI:
    STAR = "\U0001F31F"
    DIAMOND = "\U0001F539"


#######################################################################################
#                                                                                     #
#                                    TEXT WRAPPING                                    #
#                                                                                     #
#######################################################################################


def wrap_text(text: str, width: int, max_lines: int | None = None) -> list[str]:
    """Text wrapper similar to textwrap.wrap.

    This considers any characters with order >= 0x1100 as two-characters-wide, because
    commonly this is used for printing to console and consoles commonly use monospace
    fonts where 2:1 is a close approximation. (Note that 0x1100 is the first wide
    character, though there are normal-width characters after that.)

    There might be still a slight overflow in the actual display due to the
    approximation. One would need to relax ``width`` if a strict guarantee is desired.

    Parameters
    ----------
    text : str
        The text for which its representation will be wrapped.
    width : int
        The maximum width of the wrapped text. Must be at least 7.
    max_lines : int, optional
        The maximum number of lines to display. If ``None``, there is no limit. If the
        text is too long to display within ``max_lines``, it will be marked with a
        placeholder "[...]".

    Returns
    -------
    wrapped : list of str
        The list of the wrapped lines of the text.
    """
    contents = list(repr(text)[-2:0:-1])
    if max_lines is not None:
        max_lines = max(1, max_lines)
    remaining_lines = max_lines
    wrapped: list[str] = []

    # Cannot accept width < 7 due to the placeholder
    if width < 7:
        raise ValueError("Require 'width' at least 7.")

    while contents:
        # Break early if no remaining lines are left and plug in placeholder
        if remaining_lines == 0:
            lastln, n_replace, cum_length = wrapped[-1], 0, 0
            while cum_length < 6:
                n_replace += 1
                cum_length += 2 if ord(lastln[-n_replace]) >= 0x1100 else 1
            wrapped[-1] = f"{lastln[:-n_replace]} [...]"
            break

        # Create a new line and loop until the next character cannot fit in
        line, line_length = [], 0
        while contents:
            char = contents[-1]
            char_length = 2 if ord(char) >= 0x1100 else 1
            if line_length + char_length > width:
                break
            line.append(char)
            line_length += char_length
            contents.pop()
        wrapped.append("".join(line))

        # Reduce the number of remaining lines if there is a restriction
        if remaining_lines is not None:
            remaining_lines -= 1

    return wrapped


def wrap_with_prefix(
    prefix: Any,
    content: Any,
    max_lines: int | None = None,
    prefix_color: COLOR | None = None,
    content_color: COLOR | None = None,
) -> str:
    """Wraps the content with a prefix in the first line.

    Parameters
    ----------
    prefix : Any
        The prefix, taking 1/4 of the terminal width from the left.
    content : Any
        The content, taking 3/4 of the terminal width from the right.
    max_lines : int, optional
        The maximum number of lines to display. If ``None``, there is no limit. The
        prefix and content, if too long to display within ``max_lines``, will be marked
        with a placeholder "[...]".
    prefix_color : COLOR, optional
        The color of the prefix. If ``None``, do not color the prefix.
    content_color : COLOR, optional
        The color of the content. If ``None``, do not color the content.

    Returns
    -------
    wrapped : str
        The wrapped (and colored) prefix and content.
    """
    prefix_width = shutil.get_terminal_size().columns // 4 - 2  # Avoid overflow
    content_width = prefix_width * 3

    # Make sure the maximum widths are enough for the placeholder
    prefix_width = max(7, prefix_width)
    content_width = max(7, content_width)

    # Wrap the prefix and the content respectively
    prefix_lns = wrap_text(f"{prefix}", width=prefix_width, max_lines=max_lines)
    content_lns = wrap_text(f"{content}", width=content_width, max_lines=max_lines)
    prefix_nlns, content_nlns = len(prefix_lns), len(content_lns)

    # Combine the prefix and the content lines
    lns = []
    for i in range(max(prefix_nlns, content_nlns)):
        prefix_ln = (
            colored("│", prefix_color) + " " * (prefix_width - 1)
            if i >= prefix_nlns
            else colored(f"{prefix_lns[i]:<{prefix_width}}", prefix_color)
        )
        content_ln = "" if i >= content_nlns else colored(content_lns[i], content_color)
        lns.append(f"{prefix_ln}  {content_ln}")
    return "\n".join(lns)


#######################################################################################
#                                                                                     #
#                                  OTHER FORMATTING                                   #
#                                                                                     #
#######################################################################################


def format_data_item(data_item: DataItemType) -> str:
    """Format a data item into a nice printout.

    Parameters
    ----------
    data_item : DataItemType
        The data item to format.

    Returns
    -------
    formatted_data_item : str
        The formatted data item.
    """
    if isinstance(data_item, list):
        return f" {EMOJI.DIAMOND} ".join(data_item)
    if isinstance(data_item, (dict, pd.Series)):
        return f" {EMOJI.DIAMOND} ".join([f"[{k}] {v}" for k, v in data_item.items()])
    raise TypeError(f"Invalid data item of type {type(data_item)}.")
