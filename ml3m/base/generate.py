import asyncio
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Generator

import pandas as pd

from .._async import AsyncRunner
from .._color import COLOR, colored
from .._logging import manage_timed_logs
from .._paths import ensure_path, validate_path
from .._typing import DataItemType, DatasetFormat, LoggingMode


class ResponseGenerator:
    """Generate responses and combine with the original dataset.

    Parameters
    ----------
    orig_dataset : str or pathlib.Path
        The absolute path to the original dataset.
    dataset : str or pathlib.Path
        The absolute path to the result dataset. All information in the original
        dataset will be preserved while the responses will be appended.
    query_func : Callable
        The function that queries a model given a data item and outputs the model
        response. The input parameter should be a :class:`pandas.Series`, a list, or a
        dictionary, depending on ``format``. The output should be a single string
        representing the model response.
    response_name : str
        The key or column name to use for the response. This should *not* be a key or
        column name that already exists in the dataset. Be extremely careful since
        there will be *no* warning or exception raised on this.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    n_workers : int, default=1
        The number of workers. If only one worker, the dataset will be processed
        sequentially. Otherwise it will be asynchronously parallelized with the
        specified number of workers.
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
        orig_dataset: str | Path,
        dataset: str | Path,
        query_func: Callable[[DataItemType], str],
        response_name: str,
        *,
        fmt: DatasetFormat = "jsonl",
        n_workers: int = 1,
        logging_mode: LoggingMode = "all",
        verbose: int = 1,
    ) -> None:
        self.orig_dataset = orig_dataset
        self.dataset = dataset
        self.query_func = query_func
        self.fmt = fmt
        self.response_name = response_name
        self.n_workers = n_workers
        self.logging_mode = logging_mode
        self.verbose = verbose

        # Validate the arguments
        validate_path(self.orig_dataset)
        if not callable(self.query_func):
            raise ValueError("query_func must be a callable.")
        if self.fmt not in ["jsonl", "json", "csv"]:
            raise ValueError(
                f"Invalid fmt: '{self.fmt}'; must be one of 'jsonl', 'json', and "
                "'csv'."
            )

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
        source: str | Path
        using_dataset: bool = True

        if not os.path.exists(self.dataset):
            ensure_path(self.dataset)
            source = self.orig_dataset
            using_dataset = False
        else:
            source = self.dataset

        # Load the all data from the best source
        self._all_data: list | pd.DataFrame
        if self.fmt == "jsonl":
            with open(source, "r", encoding="utf-8") as f:
                self._all_data = [json.loads(line) for line in f]
        elif self.fmt == "json":
            with open(source, "r", encoding="utf-8") as f:
                self._all_data = json.load(f)
        else:  # self.fmt == "csv"
            self._all_data = pd.read_csv(source)

        # Yield the indices and corresponding data items
        if using_dataset and not overwrite:
            if self.fmt == "jsonl" or self.fmt == "json":
                for i, data_item in enumerate(self._all_data):
                    if self.response_name not in data_item or pd.isna(
                        data_item[self.response_name]
                    ):
                        yield i, data_item
            else:  # self.format == "csv"
                if self.response_name not in self._all_data.columns:
                    for i, data_item in self._all_data.iterrows():
                        yield i, data_item
                else:
                    for i, data_item in self._all_data[
                        self._all_data[self.response_name].isna()
                    ].iterrows():
                        yield i, data_item
        else:
            if self.fmt == "jsonl" or self.fmt == "json":
                for i, data_item in enumerate(self._all_data):
                    yield i, data_item
            else:  # self.format == "csv"
                for i, data_item in self._all_data.iterrows():
                    yield i, data_item

    def generate(self, *, overwrite: bool = False) -> bool:
        """Generate responses and combine with the original dataset.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the responses if some already exist, specified by
            ``response_name``.

        Returns
        -------
        completed : bool
            Whether the task has been completed.
        """
        mlog_path = manage_timed_logs(prefix=type(self).__name__)

        async def process_func(
            item: tuple[int, DataItemType],
            addtlks: list[asyncio.Lock] | None = None,
            **kwargs,
        ) -> Coroutine[
            Any, Any, tuple[tuple[int, str], str, None] | tuple[None, None, str]
        ]:
            """The process function required for the asynchronous runner."""
            i, data_item = item
            response: str | None = None
            norm_msg: str | None = None
            err: Exception | None = None
            err_trace: str | None = None

            # Handle all exceptions
            try:
                response = self.query_func(data_item)
                norm_msg = f"Item.{i:<10} {response:.40s}"
            except Exception as e:
                err, err_trace = e, traceback.format_exc()

            # Write the log on demand
            if (
                self.logging_mode == "failed"
                and response is None
                or self.logging_mode == "all"
            ):
                mlog_item = {
                    "time": str(datetime.now()),
                    "index": i,
                    "response": response,
                    "norm_msg": norm_msg,
                    "err_msg": err_trace,
                }
                async with addtlks[0]:
                    with open(mlog_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(mlog_item, ensure_ascii=False) + "\n")

            # Return the information based on success or failure
            if response is not None:
                return (i, response), norm_msg, None
            return None, None, f"Item.{i:<10} {type(err).__name__}: {err!s:.30s}"

        # Activate the asynchronous runner (sequential mode if only one worker)
        runner = AsyncRunner(
            items=self._yield_dataset(overwrite=overwrite),
            worker_kwargs=[{} for _ in range(self.n_workers)],
            process_func=process_func,
            n_locks=1,
            verbose=self.verbose,
        )
        results: list[tuple[int, str]]
        results, failed = runner.run()
        completed = len(failed) == 0

        # Update the file with the obtained results; all items must be updated,
        # including the failing ones, which should be marked as None
        result_responses = dict(results)
        if self.fmt == "jsonl" or self.fmt == "json":
            for i, item in enumerate(self._all_data):  # list of dict or list of list
                response = result_responses[i] if i in result_responses else None
                if isinstance(item, dict):
                    item[self.response_name] = response
                elif isinstance(item, list):
                    self._all_data[i] = {"data": item, self.response_name: response}
                else:
                    raise ValueError(
                        f"Each data item must be a list or a dictionary; got '{item}' "
                        f"of type '{type(item)}'."
                    )
            with open(self.dataset, "w", encoding="utf-8") as f:
                if self.fmt == "jsonl":
                    for data_item in self._all_data:
                        f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
                else:  # self.fmt == "json"
                    json.dump(self._all_data, f, ensure_ascii=False, indent=4)
        else:  # self.fmt == "csv"
            for i in self._all_data.index:  # pd.DataFrame
                response = result_responses[i] if i in result_responses else None
                self._all_data.at[i, self.response_name] = response
            self._all_data.to_csv(self.dataset, index=False)

        # Summarize the save location (and possibly log location)
        print(colored("Dataset can be found at:", COLOR.GREEN))
        print(os.path.abspath(self.dataset))
        if self.logging_mode != "none" and os.path.exists(mlog_path):
            print(colored("Execution log can be found at:", COLOR.GREEN))
            print(os.path.abspath(os.path.abspath(mlog_path)))
        return completed
