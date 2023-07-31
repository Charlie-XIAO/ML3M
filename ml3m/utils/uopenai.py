import asyncio
import json
import os
import traceback
import warnings
from pathlib import Path
from typing import Any, Coroutine, Literal

import openai

from .._typing import InputType


class OpenAIConfig:
    """OpenAI configuration.
    
    Attributes
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


def get_openai_config(
        config_path: str | Path | None = None,
        on_error: Literal["raise", "warn", "ignore"] = "raise"
    ) -> list[OpenAIConfig]:
    """Get the configurations for OpenAI.

    Parameters
    ----------
    config_path : str or pathlib.Path, optional
        If ``config_path`` is ``None``, this will read the environment variables.
        Otherwise, this will read from the specified ``config_path``, provided as
        an absolute path. It is recommended to use ``os.path.join``.
    on_error : {"raise", "warn", "ignore"}, default="raise"
        Whether to raise, warn, or ignore when meeting bad configurations. Bad
        configurations include missing keys in the configuration file, or API key not
        having a matching API base in the environment variables.
    
    Returns
    -------
    openai_configs : list of OpenAIConfig
        The list of OpenAI configuration objects.
    """
    openai_configs: list[OpenAIConfig] = []
    source: str

    def warn_or_raise(msg, on_error):
        if on_error == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        elif on_error == "raise":
            raise ValueError(msg)

    # Read from the configuration file
    if config_path is not None:
        abs_config_path = os.path.abspath(config_path)
        with open(abs_config_path, "r", encoding="utf-8") as f:
            configs: list[dict[str, str]] = json.load(f)
        for config in configs:
            if "key" in config:
                openai_configs.append(
                    OpenAIConfig(
                        key=config["key"],
                        n_workers=int(config["n_workers"]),
                        base=config.get("base", None),
                    )
                )
            else:
                warn_or_raise(
                    f"Key not found in the configuration item {config}", on_error
                )
        source = f"configuration file at `{abs_config_path}`"

    # Read from the environment variables
    else:
        for k, v in os.environ.items():
            if k.startswith("OPENAI_API_KEY_"):
                key_id = k[15:]
                desired_base = f"OPENAI_API_BASE_{key_id}"
                desired_n_workers = f"OPENAI_API_N_WORKERS_{key_id}"

                # Try to get base corresponding to key or globally
                cur_base: str | None = None
                if desired_base in os.environ:
                    cur_base = os.environ[desired_base]
                elif "OPENAI_API_BASE" in os.environ:
                    cur_base = os.environ["OPENAI_API_BASE"]
                else:
                    warn_or_raise(
                        f"No matching base for `{k}` from the environment variables. "
                        f"Set either `OPENAI_API_BASE` or `{desired_base}`.",
                        on_error,
                    )

                # Try to get n_workers corresponding to key or globally
                cur_n_workers: int | None = None
                if desired_n_workers in os.environ:
                    cur_n_workers = int(os.environ[desired_n_workers])
                elif "OPENAI_API_N_WORKERS" in os.environ:
                    cur_n_workers = int(os.environ["OPENAI_API_N_WORKERS"])
                else:
                    warn_or_raise(
                        f"No matching n_workers for `{k}` from the environment "
                        f"variables. Set either `OPENAI_API_N_WORKERS` or "
                        f"`{desired_n_workers}`.",
                        on_error,
                    )

                # Append the config if both base and n_workers are found
                openai_configs.append(
                    OpenAIConfig(key=v, n_workers=cur_n_workers, base=cur_base)
                )
        source = "environment variables"

    # Validate the configurations to make sure there are no duplicate keys and the
    # configurations are not empty
    if len(openai_configs) == 0:
        raise ValueError(f"No valid OpenAI configuration found. Check the {source}.")
    key_set = set(openai_config.key for openai_config in openai_configs)
    if len(key_set) != len(openai_configs):
        raise ValueError(f"Duplicate keys found. Check the {source}.")
    return openai_configs


async def _openai_chatcompletion(
    msgs: list[dict[str, str]],
    openai_config: OpenAIConfig,
    timeout: float = 60,
    model: str = "gpt-3.5-turbo",
    err_verbose : int = 1,
    **kwargs,
) -> Coroutine[Any, Any, tuple[str | None, dict | None, str | None]]:
    """OpenAI asynchronous ChatCompletion.

    Parameters
    ----------
    msgs : list of dict
        A list of messages comprising the conversation so far. See also
        https://platform.openai.com/docs/api-reference/chat/create
    openai_config : OpenAIConfig
        The OpenAI configuration object used for the current query.
    timeout : float, default=60
        The timeout in seconds. This is not the OpenAI timeout, but the timeout for
        cancelling the worker task.
    model : str, default="gpt-3.5-turbo"
        The ID of the model to use, must be one of the available OpenAI models that
        support the ChatCompletion API. See also
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    err_verbose : int, default=1
        The verbosity level of the error message (if exists). For level 0, only the
        exception type will be included. For level 1, the exception message will also
        be included. For level higher than 2, the full stack trace will be included.

    Returns
    -------
    reply : str or None
        The model reply. ``None`` if any exception has occurred during the querying.
    usage : dict or None
        The token usage, with keys "prompt_tokens", "completion_tokens", and
        "total_tokens". ``None`` if any exception has occurred during the querying,
        meaning that token is not consumed.
    errmsg : str or None
        The error message, if exists. If no exception has occurred but the model
        response stopped for an unexpected reason, ``errmsg`` will state that reason
        while ``reply`` and ``usage`` are both not ``None``. If any exception has
        occurred during the querying, ``errmsg`` will reflect the exception, of which
        the verbosity depends on ``err_verbose``.
    """
    try:
        completion = await asyncio.wait_for(
            openai.ChatCompletion.acreate(
                model=model,
                messages=msgs,
                api_key=openai_config.key,
                api_base=openai_config.base,
                **kwargs,
            ),
            timeout=timeout,
        )
        finish_reason: str = completion["choices"][0]["finish_reason"]
        reply: str = completion["choices"][0]["message"]["content"]
        usage: dict = completion["usage"]
        errmsg = None if finish_reason == "stop" else f"Finished with {finish_reason}"
    except Exception as e:
        reply, usage = None, None
        errmsg = traceback.format_exc() if err_verbose >= 2 else (
            f"{type(e).__name__}: {e}" if err_verbose == 1 else type(e).__name__
        )
    return reply, usage, errmsg
