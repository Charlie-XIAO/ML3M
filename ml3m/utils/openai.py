from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class _OpenAIConfig:
    """OpenAI configuration.

    Parameters
    ----------
    key : str
        The OpenAI API key.
    n_workers : int
        The maximum number of workers to parallelize using this OpenAI API key.
    base : str or None
        The OpenAI API base. ``None`` to use the default base.
    """

    def __init__(self, key: str, n_workers: int, base: str | None = None):
        self.key = key
        self.n_workers = n_workers
        self.base = base or "https://api.openai.com/v1"

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} <\n    \033[92mkey\033[0m {self.key},\n"
            f"    \033[92mbase\033[0m {self.base},\n    \033[92mn_workers\033[0m "
            f"{self.n_workers},\n>"
        )

    def __repr__(self) -> str:
        return str(self)


def get_openai_config(config_path: str | Path) -> list[_OpenAIConfig]:
    """Get the configurations for OpenAI.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The absolute path to the configuration file.

    Returns
    -------
    openai_configs : list
        The list of OpenAI configuration objects.

    Examples
    --------
    Assume that the configuration file ``.config/openai.json`` looks like the
    following:

    .. code-block::

        [
            {
                "key": "sk-xxx1",
                "base": null,
                "n_workers": 30
            },
            {
                "key": "sk-xxx2",
                "base": "http://127.0.0.1/",
                "n_workers": 5
            }
        ]

    >>> get_openai_config(".config/openai.json")  # doctest: +SKIP
    [_OpenAIConfig <
        key sk-xxx1,
        base https://api.openai.com/v1,
        n_workers 30,
    >, _OpenAIConfig <
        key sk-xxx2,
        base http://127.0.0.1/,
        n_workers 5,
    >]
    """
    abs_config_path = os.path.abspath(config_path)
    with open(abs_config_path, "r", encoding="utf-8") as f:
        configs: list[dict[str, str]] = json.load(f)
    openai_configs = [
        _OpenAIConfig(
            key=config["key"],
            n_workers=int(config["n_workers"]),
            base=config.get("base", None),
        )
        for config in configs
    ]

    # Validate the loaded configurations (not exhaustive)
    if len(openai_configs) == 0:
        raise ValueError("No valid OpenAI configuration found.")
    key_set = {openai_config.key for openai_config in openai_configs}
    if len(key_set) != len(openai_configs):
        raise ValueError("Duplicate OpenAI API keys found in the configuration file.")
    return openai_configs
