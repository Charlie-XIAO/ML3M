"""This file is for customizing exceptions."""


from __future__ import annotations

from typing import Any


class InvalidParameterError(Exception):
    """Exception for invalid parameters.

    Parameters
    ----------
    param_name: str
        The name of the invalid parameter.
    actual : Any, optional
        The actual value of the invalid parameter. If ``None``, this is not revealed in
        the error message.
    reason : str, optional
        The reason why the parameter is invalid. If ``None``, this is not revealed in
        the error message.
    """

    def __init__(
        self, param_name: str, *, actual: Any | None = None, reason: str | None = None
    ) -> None:
        msg = f"Invalid '{param_name}'"
        if actual is not None:
            msg += f"; got '{actual}'"
        if reason is not None:
            msg += f"; {reason}"
        msg += "."
        super().__init__(msg)


class ScoringError(Exception):
    """Exception when scoring.

    ``ScoringError`` can happen mainly in the following cases:

    - In an evaluator, the score obtained is not a real value, or the scores obtained
      are not represented as a dictionary with real values.
    - In an evaluator, there are multiple subjects assigned, but only a single score
      obtained.
    - In an evaluator, there exist assigned subjects that are not found in the obtained
      scores.
    """
