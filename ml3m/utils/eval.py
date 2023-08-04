import asyncio
import json
import os
import traceback
import warnings
from functools import partial
from numbers import Real
from pathlib import Path
from typing import Any, Coroutine, Generator, Literal, NoReturn

import pandas as pd
from tqdm import tqdm

from .openai import _openai_chatcompletion, get_openai_config, OpenAIConfig
from .._docstring import format_docstring
from .._logging import _manage_timed_logs
from .._typing import DataItemType, DatasetFormat


_DOCSTRINGS = {}

_DOCSTRINGS["prereq"] = r"""
    To use this evaluator, a dataset of a set of inputs, expected responses, and actual
    responses need to be prepared. The dataset can take the following formats:

    - ``.jsonl``: Each line will be read by :func:`json.loads`, either as a list or as
      a dictionary. This is to be handled in :meth:`{method}`.

    - ``.json``: The whole file will be read by :func:`json.load`, and it should be a
      json array (and read as a list). Each item in the list will be treated in the
      same way as each line in the ``.jsonl`` format. It can be either a list or a
      dictionary, to be handled in :meth:`{method}`.

    - ``.csv``: The whole file will be read by :func:`pandas.read_csv` as a
      :class:`pandas.DataFrame`. Each row is to be handled in
      :meth:`{method}`.
"""


@format_docstring(
    prereq=_DOCSTRINGS["prereq"].format(method="BaseEvaluator._get_score")
)
class BaseEvaluator:
    """Base evaluator class.
    
    This class is meant to be subclassed. The methods that must be overridden include:

    - :meth:`BaseEvaluator._get_score`

    {prereq}

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the ``evaluate`` method.
    format : {{"jsonl", "json", "csv"}}, default="jsonl"
        The format of ``dataset``, as specified above.
    """
    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        *,
        format: DatasetFormat = "jsonl",
    ) -> None:
        self.dataset = dataset
        self.save_path = save_path
        self.format = format

        # Validate the paths and format
        if not os.path.exists(self.dataset):
            raise ValueError(f"Dataset not found at {os.path.abspath(self.dataset)}")
        if self.format not in ["jsonl", "json", "csv"]:
            raise ValueError(f"Invalid format: {self.format}")
        self.sync()

    def _get_score(data_item: DataItemType) -> Real | dict[Any, Real]:
        """Evaluate a data item and obtain its score(s).

        :meta public:

        Parameters
        ----------
        data_item : DataItemType
            The data item. If ``format="jsonl"``, this is one line of ``dataset``. If
            ``format="json"``, this is one item of the json array loaded from
            ``dataset``. If ``format="csv"``, this is one row of the
            :class:`pandas.DataFrame` loaded from ``dataset``.

        Returns
        -------
        scores : numbers.Real or dict
            The evaluated scores, either a single score or a dictionary of subject-
            score pairs.

        Notes
        -----
        This method is not implemented and must be overridden in subclasses.
        """
        raise NotImplementedError

    def evaluate(self, *, overwrite: bool = False) -> None:
        """Evaluate the specified dataset.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the data in ``save_path``. If ``False``, the
            evaluation will be built upon existing data in ``save_path``, otherwise
            all data will be evaluated are existing data will be overwritten.
        """
        if overwrite:
            self._df = pd.DataFrame(columns=["i"], dtype=pd.Int64Dtype)
        overview = list(self._yield_data_in_iteration())
        if not overview:
            print("The evaluation has been fully completed.")
            return

        # Run the evaluation on each data item
        progbar = tqdm(total=len(overview))
        result_scores: dict[int, dict[Any, Real]] = {}
        for i, data_item in self._yield_data(overwrite=overwrite):
            eval_scores = self._check_scores(self._get_score(data_item))
            result_scores[i] = eval_scores
        progbar.close()

        # Update the obtained data and write the DataFrame
        print("Updating results...", end=" ", flush=True)
        new_df = pd.DataFrame(result_scores).T.reset_index(names="i")
        self._df = pd.concat([self._df, new_df])
        missing_data = self._df.isna().any()
        if missing_data.any():
            warnings.warn(
                "\033[93mUnexpected missing values detected in the columns "
                f"{list(missing_data[missing_data].index)}\033[0m",
                UserWarning,
                stacklevel=2,
            )
        self._df.convert_dtypes().sort_values(by=["i"]).to_csv(
            self.save_path, index=False
        )
        print(f"done, available at:\n{os.path.abspath(self.save_path)}")

    def sync(self) -> None:
        """Sync up with the results in the save path.

        This method should be called whenever the file at ``save_path`` is modified yet
        one still uses the original evaluator instance.
        """
        abs_save_path = os.path.abspath(self.save_path)
        if not os.path.exists(self.save_path):
            print(
                f"\033[93mSave path not found at {abs_save_path}; forcefully created"
                "\033[0m",
            )
            directories, _ = os.path.split(abs_save_path)
            if not os.path.exists(directories):
                os.makedirs(directories)
            self._df = pd.DataFrame(columns=["i"], dtype=pd.Int64Dtype)
            self._df.to_csv(self.save_path, index=False)
        else:
            try:
                self._df = pd.read_csv(self.save_path)
            except Exception as e:
                raise type(e)(
                    "Failed to load as a DataFrame from ``save_path``\nPath: "
                    f"{abs_save_path}\nDetails: {e}"
                )

        # Validate the loaded DataFrame (not exhaustive)
        self._df = self._df.convert_dtypes()
        if "i" not in self._df.columns:
            raise ValueError(
                "DataFrame loaded from ``save_path`` does not have the column 'i'\n"
                f"Path: {abs_save_path}"
            )
        if len(self._df) > 0 and (
            not pd.api.types.is_integer_dtype(self._df["i"].dtype)
            or not all(
                pd.api.types.is_numeric_dtype(dtype) for dtype in self._df.dtypes
            )
        ):
            raise ValueError(
                "DataFrame loaded from ``save_path`` has wrong dtype; the index "
                "column 'i' is required to be integer dtype, and the other columns "
                "representing scoring subjects are required to be real dtype\nPath: "
                f"{abs_save_path}"
            )

    def _yield_data(
        self, overwrite: bool = False
    ) -> Generator[tuple[int, DataItemType], Any, None]:
        """Yield the indices and data items to be done.

        Yield
        -----
        i : int
            The index of the data item. It is the index of the line if
            ``format="jsonl"``, the index in the json array if ``format="json"``, or
            the index of the row if ``format="csv"``.
        data_item : DataItemType
            The data item.
        """
        existing_indices: list = []
        if not overwrite:
            existing_indices = list(self._df.loc[:, "i"])

        # Yield indices and corresponding data items, skipping existing ones
        if self.format == "jsonl":
            with open(self.dataset, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i not in existing_indices:
                        yield i, json.loads(line)
        elif self.format == "json":
            with open(self.dataset, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            for i, item in enumerate(all_data):
                if i not in existing_indices:
                    yield i, item
        elif self.format == "csv":
            data_df = pd.read_csv(self.dataset)
            for i, row in data_df.iterrows():
                if i not in existing_indices:
                    yield i, row
        else:
            raise ValueError(f"Invalid format: {self.format}")

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
            return scores
        elif isinstance(scores, dict):
            bad_item = next(
                (
                    (subject, score) for subject, score in scores.items()
                    if not isinstance(score, Real) or pd.isna(score)
                ),
                None,
            )
            if bad_item is not None:
                raise TypeError(
                    "The scores are expected to be a real number or a dictionary with "
                    f"real values, got ``{scores}`` of type dict but there exists "
                    f"{bad_item[0]}: {bad_item[1]} of type ``{type(bad_item[1])}``."
                )
            else:
                return scores
        else:
            raise TypeError(
                "The scores are expected to be a real number or a dictionary with "
                f"real values, got ``{scores}`` of type ``{type(scores)}`` instead."
            )


@format_docstring(
    prereq=_DOCSTRINGS["prereq"].format(method="BaseOpenAIEvaluator._prompt")
)
class BaseOpenAIEvaluator(BaseEvaluator):
    """Base evaluator class via OpenAI.

    This class is meant to be subclassed. The methods that must be overriden include:

    - :meth:`BaseOpenAIEvaluator._prompt`
    - :meth:`BaseOpenAIEvaluator._extract_scores`

    {prereq}

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    openai_config : str or pathlib.Path
        The absolute path to the OpenAI configuration file.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the ``evaluate`` method.
    format : {{"jsonl", "json", "csv"}}, default="jsonl"
        The format of ``dataset``, as specified above.
    n_iter : int, default=3
        The maximum number of iterations if OpenAI querying failed on any data item.
    timeout : float, default=60
        The timeout in seconds. This is not the OpenAI timeout, but the timeout for
        cancelling the worker tasks.
    model : str, default="gpt-3.5-turbo"
        The ID of the model to use, must be one of the available OpenAI models that
        support the ChatCompletion API. See also
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    verbose : int, default=1
        The verbosity level of OpenAI querying printout. For level 0, only a progress
        bar will be displayed. For level 1, the errored queries will also be displayed.
        For level higher than 2, all queries will be displayed. Regardless of the
        verbosity level, the full log will be written, except that the verbosity of
        exceptions will depend on ``err_verbose``.
    err_verbose : int, default=1
        The verbosity level of the error message when writing logs. For level 0, only
        the exception type will be included. For level 1, the exception message will
        also be included. For level higher than 2, the full stack trace will be
        included. Regardless of the ``err_verbose``, verbosity level 0 will be used in
        printout of error messages.
    """
    def __init__(
        self,
        dataset: str | Path,
        openai_config: str | Path,
        save_path: str | Path,
        *,
        format: DatasetFormat = "jsonl",
        n_iter: int = 3,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        verbose: int = 1,
        err_verbose: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            format=format,
        )
        self.openai_config = openai_config
        self.n_iter = n_iter
        self.timeout = timeout
        self.model = model
        self.verbose = verbose
        self.err_verbose = err_verbose

        # Validate the paths
        if not os.path.exists(self.openai_config):
            raise ValueError(
                "OpenAI configuration not found at "
                f"{os.path.abspath(self.openai_config)}"
            )

        # Synchronization locks
        self._tqdmlk = asyncio.Lock()  # For tqdm progress bar update
        self._mloglk = asyncio.Lock()  # For writing log of model responses
        self._mainlk = asyncio.Lock()  # For collecting data

    def _prompt(self, data_item: DataItemType) -> tuple[str, str]:
        """Return the prompt for evaluation.

        :meta public:

        Parameters
        ----------
        data_item : DataItemType
            The data item. If ``format="jsonl"``, this is one line of ``dataset``. If
            ``format="json"``, this is one item of the json array loaded from
            ``dataset``. If ``format="csv"``, this is one row of the
            :class:`pandas.DataFrame` loaded from ``dataset``.

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

        This method should properly raise exceptions if the scores cannot be properly
        extracted from the replies. This way the exceptions can be caught, and the
        current data item will be considered a failure. Either it will be re-evaluated
        in the following iterations, or it will be left unevaluated.
        
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
        This method is not implemented and must be overridden in subclasses.
        """
        raise NotImplementedError

    def evaluate(
        self, *, overwrite: bool = False, skip_openai_api_cfm: bool = False
    ) -> None:
        """Evaluate the specified dataset.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the data in ``save_path``. If ``False``, the
            evaluation will be built upon existing data in ``save_path``, otherwise
            all data will be evaluated are existing data will be overwritten.
        skip_openai_api_cfm : bool, default=False
            Whether to skip the confirmation message that notifies possible OpenAI API
            usage. Set to ``True`` to silence the confirmation message. The default is
            ``False`` just in case that someone is not aware.
        """
        openai_configs = get_openai_config(self.openai_config)
        print(f"{len(openai_configs)} OpenAI keys detected:")
        for openai_config in openai_configs:
            print(openai_config)
        if not skip_openai_api_cfm:
            cfm = input(
                "\033[93mThis message is to notify you that the method "
                f"``{type(self).__name__}.evalute`` may consume OpenAI tokens of your "
                "account(s). If you are aware of the possible consumption, press "
                "Enter to continue. You can silence this confirmation message by "
                "specifying ``skip_open_api_cfm=True``.\033[0m"
            )
            if cfm != "":
                return

        # Check if task is already completed
        if overwrite:
            self._df = pd.DataFrame(columns=["i"], dtype=pd.Int64Dtype)
        self._yield_data_in_iteration = partial(self._yield_data, overwrite=overwrite)
        if not list(self._yield_data_in_iteration()):
            print("The evaluation has been fully completed.")
            return

        # Activate the main event loop
        mlog_path = _manage_timed_logs("openai", keep=10)
        for it in range(self.n_iter):
            asyncio.run(
                self._mainloop(
                    it=it, mlog_path=mlog_path, openai_configs=openai_configs
                )
            )

        # Write the latest updated DataFrame
        print("Updating results...", end=" ", flush=True)
        self._df.convert_dtypes().sort_values(by=["i"]).to_csv(
            self.save_path, index=False
        )
        print(f"done, available at:\n{os.path.abspath(self.save_path)}")

    async def _execute(
        self,
        *,
        queue: asyncio.Queue[tuple[int, DataItemType]],
        shared_resources: list[
            tuple[int, dict[Any, Real], Literal[True]]
            | tuple[int, DataItemType, Literal[False]]
        ],
        openai_config: OpenAIConfig,
        mlog_path : str | Path,
        progbar: tqdm,
        it_id: int,
        worker_id: int,
        openai_api_id: int,
    ) -> Coroutine[Any, Any, NoReturn]:
        """Execution task processing a data item.

        Parameters
        ----------
        queue : asyncio.Queue
            The asynchronous queue held by the main event loop.
        shared_resources : list
            The shared resources for storing results.
        openai_config : OpenAIConfig
            The OpenAI configuration object used for the current query.
        mlog_path : str or pathlib.Path
            The path for the log of OpenAI model responses.
        progbar : tqdm.tqdm
            The progress bar for updating held by the main event loop.
        it_id : int
            The id of the current iteration.
        worker_id : int
            The id of the worker task.
        openai_api_id : int
            The id of the OpenAI API.
        """
        while True:
            i, data_item = await queue.get()
            sys_msg, eval_prompt = self._prompt(data_item)

            # Query via OpenAI asynchronous API
            reply: str | None
            usage: dict | None
            errmsg: str | None
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": eval_prompt},
            ] if sys_msg else [{"role": "user", "content": eval_prompt}]
            reply, usage, errmsg = await _openai_chatcompletion(
                msgs=messages,
                openai_config=openai_config,
                timeout=self.timeout,
                model=self.model,
                err_verbose=self.err_verbose,
            )

            # Try to extract the scores from the reply, otherwise store the error
            eval_scores: dict[Any, Real] | None = None
            formatted_err: str | None = None
            mlog_item = {
                "index": i,
                "worker": worker_id,
                "kwds": {"dataset": self.dataset, "save_path": self.save_path},
                "api_id": openai_api_id,
                "api_key": openai_config.key,
                "reply": reply,
                "usage": usage,
                "errmsg": errmsg,
            }
            if errmsg is None:
                try:
                    scores = self._extract_scores(reply)
                    eval_scores = self._check_scores(scores)
                except Exception as e:
                    formatted_err = type(e).__name__
                    if self.err_verbose >= 2:
                        mlog_item["errmsg"] = traceback.format_exc()
                    elif self.err_verbose == 1:
                        mlog_item["errmsg"] = f"{type(e).__name__}: {e}"
                    else:
                        mlog_item["errmsg"] = type(e).__name__
            else:
                formatted_err = "Model error, please check the log"

            # Print to console depending on verbosity level
            prefix = f"[{worker_id:03d}::{openai_api_id:03d} > Index.{i}, It.{it_id}]"
            if eval_scores is not None and self.verbose >= 2:
                scores_msg = " ".join(
                    [f"\033[92m{k}\033[0m {v}" for k, v in eval_scores.items()]
                )
                async with self._tqdmlk:
                    tqdm.write(f"{prefix:<30} {scores_msg}")
            elif eval_scores is None and self.verbose >= 1:
                async with self._tqdmlk:
                    tqdm.write(f"{prefix:<30} \033[31m{formatted_err}\033[0m")

            # Store the model response log
            async with self._mloglk:
                with open(mlog_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(mlog_item, ensure_ascii=False) + "\n")

            # Collect the result, update progress bar, and mark task as done
            async with self._tqdmlk:
                progbar.update(1)
            async with self._mainlk:
                shared_resources.append(
                    (i, eval_scores, True) if eval_scores is not None else (
                        i, data_item, False
                    )
                )
            queue.task_done()

    async def _mainloop(
        self, *, it: int, mlog_path: str | Path, openai_configs: list[OpenAIConfig],
    ) -> Coroutine[Any, Any, None]:
        """Main event loop for asynchronous querying.

        Parameters
        ----------
        it : int
            The id of the current iteration.
        mlog_path : str or pathlib.Path
            The path to the model response log.
        openai_configs : list of OpenAIConfig
            The OpenAI configurations.
        """
        queue: asyncio.Queue[tuple[int, DataItemType]] = asyncio.Queue()
        n_items = 0
        for item in self._yield_data_in_iteration():
            queue.put_nowait(item)
            n_items += 1
        if n_items == 0:
            print("The evaluation has been fully completed.")
            return

        # Create worker tasks to process the queue asychronously
        print(f"### Iteration {it}")
        wid = 0
        tasks: list[asyncio.Task] = []
        shared_resources: list[
            tuple[int, dict[Any, Real], Literal[True]]
            | tuple[int, DataItemType, Literal[False]]
        ] = []
        progbar = tqdm(total=n_items)
        for openai_api_id, openai_config in enumerate(openai_configs):
            for _ in range(int(openai_config.n_workers)):
                tasks.append(
                    asyncio.create_task(
                        self._execute(
                            queue=queue,
                            shared_resources=shared_resources,
                            openai_config=openai_config,
                            mlog_path=mlog_path,
                            progbar=progbar,
                            it_id=it,
                            worker_id=wid,
                            openai_api_id=openai_api_id,
                        )
                    )
                )
                wid += 1
        async with self._tqdmlk:
            tqdm.write(f"{wid} workers utilized as configured for {n_items} data items")

        # Wait until the queue is fully processed and collect the results
        await queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        progbar.close()

        # Collect failed items (if exist) and print a brief summary
        print("Collecting results...", end=" ", flush=True)
        result_scores: dict[int, dict[Any, Real]] = {}
        todo_items: list[tuple[int, DataItemType]] = []
        for i, result, passed in shared_resources:
            if passed:
                result_scores[i] = result
            else:
                todo_items.append((i, result))
        if todo_items:
            print(f"\033[31m{len(todo_items)} failed\033[0m among all {n_items} items.")
        else:
            print(f"\033[92mall {n_items} items done.\033[0m")

        # Update the obtained data but postpone writing
        new_df = pd.DataFrame(result_scores).T.reset_index(names="i")
        self._df = pd.concat([self._df, new_df])
        missing_data = self._df.isna().any()
        if missing_data.any():
            warnings.warn(
                "\033[93mUnexpected missing values detected in the columns "
                f"{list(missing_data[missing_data].index)}\033[0m",
                UserWarning,
                stacklevel=2,
            )

        # Reset the yielding function
        def yield_data_in_iteration():
            for item in todo_items:
                yield item
        self._yield_data_in_iteration = yield_data_in_iteration
