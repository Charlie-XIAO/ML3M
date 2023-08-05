import asyncio
import json
import os
import traceback
import warnings
from pathlib import Path
from typing import Any, Coroutine, Literal

import openai


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


def get_openai_config(
        config_path: str | Path | None = None,
        on_error: Literal["raise", "warn", "ignore"] = "raise"
    ) -> list[OpenAIConfig]:
    """Get the configurations for OpenAI.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The absolute path to the configuration file.
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

    def warn_or_raise(msg, on_error):
        if on_error == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        elif on_error == "raise":
            raise ValueError(msg)

    # Read from the configuration file
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

    # Validate the configurations to make sure there are no duplicate keys and the
    # configurations are not empty
    if len(openai_configs) == 0:
        raise ValueError(
            "No valid OpenAI configuration found. Check the configuration file at "
            f"`{abs_config_path}`."
        )
    key_set = set(openai_config.key for openai_config in openai_configs)
    if len(key_set) != len(openai_configs):
        raise ValueError(
            "Duplicate keys found. Check the configuration file at "
            f"`{abs_config_path}`."
        )
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
        errmsg: str
        if err_verbose >= 2:
            errmsg = traceback.format_exc()
        elif err_verbose == 1:
            errmsg = f"{type(e).__name__}: {e}"
        else:
            errmsg = type(e).__name__
    return reply, usage, errmsg
