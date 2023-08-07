import asyncio
from datetime import datetime
import json
from numbers import Real
import os
from pathlib import Path
from typing import Any, Coroutine, Generator
import warnings

import openai
import pandas as pd

from .._async import AsyncRunner
from .._color import COLOR, colored
from .._logging import manage_timed_logs
from .._paths import ensure_path, validate_path
from .._typing import DataItemType, DatasetFormat, LoggingMode
from ..utils.openai import get_openai_config


class BaseEvaluator:
    """Base evaluator class.

    This class is meant to be subclassed. The methods that must be overridden include:

    - :meth:`BaseEvaluator._aget_score`

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the ``evaluate`` method.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    workers : int or list of dict, default=1
        If ``workers`` is an integer, it will be considered the number of workers. If
        specified only one worker, the dataset will be processed sequentially, and
        otherwise it will be asynchronously parallelized. If ``workers`` is a list of
        dictionaries, the length of this list will be considered the number of workers.
        Each dictionary should be the additional keyword arguments passed to
        :meth:`BaseEvaluator._aget_score`.
    logging_mode : {"all", "failed", "none"}, default="all"
        The logging mode, whether to save the logs of all items, or only of failed
        items, or save no log.
    verbose : int, default=1
        The verbosity level of the processing. For level 0, only a progress bar will be
        displayed. For level 1, the errored items will also be displayed. For levels
        higher than 2, all items will be displayed.
    """

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        *,
        fmt: DatasetFormat = "jsonl",
        workers: int | list[dict] = 1,
        logging_mode: LoggingMode = "all",
        verbose: int = 1,
    ) -> None:
        self.dataset = dataset
        self.save_path = save_path
        self.fmt = fmt
        self.workers = workers
        self.logging_mode = logging_mode
        self.verbose = verbose

        # Validate the arguments
        validate_path(self.dataset)
        if self.fmt not in ["jsonl", "json", "csv"]:
            raise ValueError(
                f"Invalid fmt: '{self.fmt}'; must be one of 'jsonl', 'json', and "
                "'csv'."
            )
        if self.logging_mode not in ["all", "failed", "none"]:
            raise ValueError(
                f"Invalid logging_mode: '{self.logging_mode}'; must be one of 'all', "
                "'failed', and 'none'."
            )

        # Load the key arguments for workers
        self._worker_kwargs: list[dict]
        if isinstance(self.workers, int):
            if self.workers < 1:
                raise ValueError(
                    f"Invalid workers: '{workers}'; if given as integer, must be >= 1."
                )
            self._worker_kwargs = [{} for _ in range(self.workers)]
        elif isinstance(self.workers, list):
            if any(not isinstance(worker, dict) for worker in self.workers):
                raise ValueError(
                    f"Invalid workers: '{workers}'; if given as list, each element "
                    "must be a keyword dictionary."
                )
            self._worker_kwargs = self.workers

    async def _aget_score(
        self, data_item: DataItemType, **kwargs
    ) -> Coroutine[Any, Any, Real | dict[Any, Real]]:
        """Evaluate a data item and obtain its score(s).

        :meta public:

        Parameters
        ----------
        data_item : DataItemType
            The data item.
        **kwargs
            The additional keyword arguments.

        Returns
        -------
        scores : numbers.Real or dict
            The evaluated scores, either a single score or a dictionary of subject-
            score pairs.

        Notes
        -----
        This method is not implemented and must be overridden in subclasses. Note that
        this function must be defined as asynchronous, but it is okay that it does not
        await for anything.

        Moreover, it is recommended *not* to catch the exceptions that cause the
        processing of a data item to fail, since otherwise
        :meth:`BaseEvaluator.evaluate` will not realize that the data item errors out.
        """
        raise NotImplementedError

    def _sync_save_path(self, overwrite: bool = False) -> None:
        """Sync up with the results in the save path.

        This loads ``save_path`` and sets ``_result_df`` correspondingly.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to ignore existing results.
        """
        self._result_df: pd.DataFrame
        if not os.path.exists(self.save_path):
            ensure_path(self.save_path)
            self._result_df = pd.DataFrame(columns=["i"], dtype=pd.Int64Dtype)
        elif overwrite:
            self._result_df = pd.DataFrame(columns=["i"], dtype=pd.Int64Dtype)
        else:
            self._result_df = pd.read_csv(self.save_path)

    def _yield_dataset(
        self, overwrite: bool = False
    ) -> Generator[tuple[int, DataItemType], Any, None]:
        """Yield the indices and data items to be done.

        Yield
        -----
        i : int
            The index of the data item.
        data_item : DataItemType
            The data item.
        """
        existing_indices: list = []
        if not overwrite:
            existing_indices = list(self._result_df.loc[:, "i"])

        # Yield the indices and corresponding data items
        if self.fmt == "jsonl":
            with open(self.dataset, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i not in existing_indices:
                        yield i, json.loads(line)
        elif self.fmt == "json":
            with open(self.dataset, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            for i, item in enumerate(all_data):
                if i not in existing_indices:
                    yield i, item
        else:  # self.fmt == "csv"
            all_data = pd.read_csv(self.dataset)
            for i, row in all_data.iterrows():
                if i not in existing_indices:
                    yield i, row

    def _check_scores(self, scores: Real | dict[Any, Real]) -> dict[Any, Real]:
        """Check and format the scores.

        Parameters
        ----------
        scores : Real or dict
            The evaluation scores of a data item, either a single real number of a
            dictionary of subject-score pairs.

        Returns
        -------
        eval_scores : dict
            If ``scores`` is a single real number, this returns ``{"scores": scores}``.
            Otherwise, this returns ``scores`` itself.

        Raises
        ------
        TypeError
            If ``scores`` is not a real number or a dictionary with real values.
        """
        if isinstance(scores, Real) and not pd.isna(scores):
            return {"scores": scores}
        elif isinstance(scores, dict):
            bad_item = next(
                (
                    (subject, score)
                    for subject, score in scores.items()
                    if not isinstance(score, Real) or pd.isna(score)
                ),
                None,
            )
            if bad_item is not None:
                raise TypeError(
                    "The scores must be either a real number or a dictionary with "
                    f"real values; got a dictionary but there exists "
                    f"'{bad_item[0]}: {bad_item[1]}' of type '{type(bad_item[1])}'."
                )
            else:
                return scores
        else:
            raise TypeError(
                "The scores must be either a real number or a dictionary with real "
                f"values; got '{scores}' of type '{type(scores)}' instead."
            )

    def evaluate(self, *, overwrite: bool = False) -> None:
        """Evaluate the specified dataset.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the data in ``save_path``. If ``False``, the
            evaluation will be built upon existing data in ``save_path``, otherwise
            all data will be evaluated are existing data will be overwritten.
        """
        self._sync_save_path(overwrite=overwrite)
        mlog_path = manage_timed_logs(type(self).__name__)
        result_scores: dict[int, dict[Any, Real]] = {}

        async def process_func(
            item: tuple[int, DataItemType],
            addtlks: list[asyncio.Lock] | None = None,
            **kwargs,
        ) -> Coroutine[Any, Any, tuple[Any, str, str]]:
            i, data_item = item
            eval_scores: dict[Any, Real] | None = None
            norm_msg: str | None = None
            err_msg: str | None = None

            # Handle all exceptions
            try:
                scores = await self._aget_score(data_item, **kwargs)
                eval_scores = self._check_scores(scores)
                norm_msg = (
                    f"Item.{i:<10} "
                    f"{json.dumps(eval_scores, ensure_ascii=False):.40s}"
                )
            except Exception as e:
                err_msg = type(e).__name__

            # Write the log on demand
            if (
                self.logging_mode == "failed"
                and eval_scores is None
                or self.logging_mode == "all"
            ):
                mlog_item = {
                    "time": str(datetime.now()),
                    "index": i,
                    "eval_scores": eval_scores,
                    "norm_msg": norm_msg,
                    "err_msg": err_msg,
                }
                async with addtlks[0]:
                    with open(mlog_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(mlog_item, ensure_ascii=False) + "\n")
            return (
                None if eval_scores is None else (i, eval_scores),
                norm_msg,
                f"Item.{i:<10} {err_msg}",
            )

        # Activate the asynchronous runner (sequential mode if only one worker)
        runner = AsyncRunner(
            items=self._yield_dataset(overwrite=overwrite),
            worker_kwargs=self._worker_kwargs,
            process_func=process_func,
            n_locks=1,
            verbose=self.verbose,
        )
        results, _ = runner.run()
        for i, eval_scores in results:
            result_scores[i] = eval_scores

        # Update the file with the obtained results
        new_df = pd.DataFrame(result_scores).T.reset_index(names="i")
        self._result_df = pd.concat([self._result_df, new_df])
        missing_data = self._result_df.isna().any()
        if missing_data.any():
            warnings.warn(
                "Unexpected missing values detected in the columns "
                f"{list(missing_data[missing_data].index)}",
                UserWarning,
                stacklevel=2,
            )
        self._result_df.convert_dtypes().sort_values(by=["i"]).to_csv(
            self.save_path, index=False
        )

        # Summarize the save location (and possibly log location)
        print(colored("Results can be found at:", COLOR.GREEN))
        print(os.path.abspath(self.save_path))
        if self.logging_mode != "none":
            print(colored("Execution log can be found at:", COLOR.GREEN))
            print(os.path.abspath(os.path.abspath(mlog_path)))


class BaseOpenAIEvaluator(BaseEvaluator):
    """Base evaluator class via OpenAI.

    This class is meant to be subclassed. The methods that must be overriden include:

    - :meth:`BaseOpenAIEvaluator._prompt`
    - :meth:`BaseOpenAIEvaluator._extract_scores`

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the ``evaluate`` method.
    openai_config : str or pathlib.Path
        The absolute path to the OpenAI configuration file.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    timeout : float, default=60
        The timeout in seconds. This is not the OpenAI timeout, but the timeout for
        cancelling the worker tasks.
    model : str, default="gpt-3.5-turbo"
        The ID of the model to use, must be one of the available OpenAI models that
        support the ChatCompletion API. See also
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    logging_mode : {"all", "failed", "none"}, default="all"
        The logging mode, whether to save the logs of all items, or only of failed
        items, or save no log.
    verbose : int, default=1
        The verbosity level of the processing. For level 0, only a progress bar will be
        displayed. For level 1, the errored items will also be displayed. For levels
        higher than 2, all items will be displayed.
    """

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        openai_config: str | Path,
        *,
        fmt: DatasetFormat = "jsonl",
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        logging_mode: LoggingMode = "all",
        verbose: int = 1,
    ) -> None:
        self.openai_config = openai_config
        self.timeout = timeout
        self.model = model

        # Load the OpenAI configurations
        validate_path(self.openai_config)
        worker_kwargs = [
            {"api_key": config.key, "api_base": config.base}
            for config in get_openai_config(self.openai_config)
            for _ in range(config.n_workers)
        ]

        # Inherit from parent
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            fmt=fmt,
            workers=worker_kwargs,
            logging_mode=logging_mode,
            verbose=verbose,
        )

    def _prompt(self, data_item: DataItemType) -> tuple[str, str]:
        """Return the prompt for evaluation.

        :meta public:

        Parameters
        ----------
        data_item : DataItemType
            The data item.

        Returns
        -------
        sys_msg : str
            The system message for setting the role of the OpenAI model when querying
            for evaluation, e.g. a professional teacher in some field. If no system
            message is needed, this should be an empty string. See also
            https://platform.openai.com/docs/guides/gpt/chat-completions-api
            for an example of system message.
        eval_prompt : str
            The formatted evaluation prompt.

        Notes
        -----
        This method is not implemented and must be overridden in subclasses.
        """
        raise NotImplementedError

    def _extract_scores(self, reply: str) -> Real | dict[Any, Real]:
        """Extract the score(s) from the OpenAI model reply.

        :meta public:

        This method should correspond to the :meth:`BaseOpenAIEvaluator._prompt`
        method, in the sense that the formatted evaluation prompt is expected to invoke
        an *extractable* model reply, and this method should extract the score(s) from
        that reply. It can extract either a single score or a dictionary of subject-
        score pairs.

        Parameters
        ----------
        reply : str
            The OpenAI model reply, from which the score(s) will be extracted.

        Returns
        -------
        scores : numbers.Real or dict
            The extracted scores, either a single score or a dictionary of subject-
            score pairs.

        Notes
        -----
        This method is not implemented and must be overridden in subclasses. Moreover,
        it is recommended *not* to catch the exceptions that cause the extraction of
        scores to fail, since otherwise :meth:`BaseOpenAIEvaluator.evaluate` will not
        realize that the data item errors out.
        """
        raise NotImplementedError

    async def _aget_score(
        self, data_item: DataItemType, **kwargs
    ) -> Coroutine[Any, Any, Coroutine[Any, Any, Real | dict[Any, Real]]]:
        sys_msg, eval_prompt = self._prompt(data_item)
        messages = (
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": eval_prompt},
            ]
            if sys_msg
            else [{"role": "user", "content": eval_prompt}]
        )

        # Asynchronous query for OpenAI
        completion = await asyncio.wait_for(
            openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                api_key=kwargs["api_key"],
                api_base=kwargs["api_base"],
            ),
            timeout=self.timeout,
        )

        # Check whether the model has fully completed the response
        finish_reason = completion["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            raise ValueError(f"Model terminated by '{finish_reason}'")
        return self._extract_scores(completion["choices"][0]["message"]["content"])
