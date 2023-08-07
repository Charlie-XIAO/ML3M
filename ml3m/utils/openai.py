import json
import os
from pathlib import Path


class OpenAIConfig:
    """OpenAI configuration.

    Parameters
    ----------
    key : str
        The OpenAI API key.
    base : str
        The OpenAI API base.
    """

    def __init__(self, key: str, n_workers: int, base: str | None = None):
        self.key = key
        self.n_workers = n_workers
        self.base = base or "https://api.openai.com/v1"

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} <\n    \033[92mkey\033[0m {self.key},\n"
            f"    \033[92mbase\033[0m {self.base},\n    \033[92mn_workers\033[0m "
            f"{self.base},\n>"
        )

    def __repr__(self) -> str:
        return str(self)


def get_openai_config(config_path: str | Path) -> list[OpenAIConfig]:
    """Get the configurations for OpenAI.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The absolute path to the configuration file.

    Returns
    -------
    openai_configs : list of OpenAIConfig
        The list of OpenAI configuration objects.
    """
    abs_config_path = os.path.abspath(config_path)
    with open(abs_config_path, "r", encoding="utf-8") as f:
        configs: list[dict[str, str]] = json.load(f)
    openai_configs = [
        OpenAIConfig(
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
