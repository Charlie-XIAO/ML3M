from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime
from numbers import Real
from typing import TYPE_CHECKING, Any, Generator

import openai
import pandas as pd

from .._async import AsyncRunner
from .._display import COLOR, colored, format_data_item
from .._logging import manage_timed_logs
from .._paths import ensure_path, validate_path
from ..errors import InvalidParameterError, ScoringError
from ..utils.openai import get_openai_config

if TYPE_CHECKING:
    from pathlib import Path

    from .._typing import AggregateMethod, DataItemType, DatasetFormat, LoggingMode


class BaseEvaluator:
    """Base evaluator class.

    .. note::
        This class is meant to be subclassed. The methods that must be overridden
        include:

        - :meth:`BaseEvaluator._aget_score` (if to be used with multiple workers)
        - :meth:`BaseEvaluator._get_score` (if to be used with a single worker)

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the :meth:`BaseEvaluator.evaluate` method.
    subjects : list
        The subjects to evaluate. This should *strictly* correspond to how to scores
        are obtained in :meth:`BaseEvaluator._aget_score`. If the score is obtained as
        a single real value, ``subjects`` must be a list of one element and that
        element will be used as the name of that score. If the score(s) are obtained as
        a dictionary of subject-value pairs, all items in ``subjects`` must appear as
        the keys of the obtained dictionary. Any additional key will be discarded; any
        missing key will be treated as an error.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    workers : int or list of dict, default=1
        If ``workers`` is an integer, it will be considered the number of workers. If
        specified only one worker, the dataset will be processed sequentially, and
        otherwise it will be asynchronously parallelized. If ``workers`` is a list of
        dictionaries, the length of this list will be considered the number of workers.
        Each dictionary should be the additional keyword arguments passed to
        :meth:`BaseEvaluator._aget_score`. Note that if ``workers`` is an integer, no
        additional keyword arguments will be passed.
    n_iter : int, default=1
        The number of iterations for each data item. This is commonly used when one
        round of scoring is not convincing enough and the final scoring should be some
        statistics of multiple rounds of scoring.
    agg_method : {"mean", "sum", "min", "max", "mode"}, default=None
        The aggregate method to use on multiple rounds of scoring. Ignored when
        ``n_iter=1``, and otherwise this must not be ``None``.
    logging_mode : {"all", "failed", "none"}, default="all"
        The logging mode, whether to save the logs of all items, or only of failed
        items, or save no log.
    verbose : int, default=0
        The verbosity level of the processing. For negative levels, only a progress bar
        will be displayed. For level 0, the errored items will also be displayed. For
        positive levels, the all items will be displayed, and the verbosity level
        determines the number of lines to display for the message of each item.
    """

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        subjects: list,
        *,
        fmt: DatasetFormat = "jsonl",
        workers: int | list[dict] = 1,
        n_iter: int = 1,
        agg_method: AggregateMethod | None = None,
        logging_mode: LoggingMode = "all",
        verbose: int = 0,
    ) -> None:
        self.dataset = dataset
        self.save_path = save_path
        self.subjects = subjects
        self.fmt = fmt
        self.workers = workers
        self.n_iter = n_iter
        self.agg_method = agg_method
        self.logging_mode = logging_mode
        self.verbose = verbose

        # Validate the arguments
        validate_path(self.dataset)
        if not isinstance(self.subjects, list):
            raise InvalidParameterError(
                "subjects", actual=self.subjects, reason="must be a list"
            )
        if len(self.subjects) == 0:
            raise InvalidParameterError(
                "subjects", actual=self.subjects, reason="must not be empty"
            )
        if "i" in subjects:
            raise InvalidParameterError(
                "subjects",
                actual=self.subjects,
                reason="'i' is a reserved key for indexing",
            )
        if self.fmt not in ["jsonl", "json", "csv"]:
            raise InvalidParameterError(
                "fmt",
                actual=self.fmt,
                reason="must be one of 'jsonl', 'json', and 'csv'",
            )
        if self.logging_mode not in ["all", "failed", "none"]:
            raise InvalidParameterError(
                "logging_mode",
                actual=self.logging_mode,
                reason="must be one of 'all', 'failed', and 'none'",
            )
        if not isinstance(self.n_iter, int) or self.n_iter < 1:
            raise InvalidParameterError(
                "n_iter", actual=self.n_iter, reason="must be an integer >= 1"
            )
        if n_iter > 1 and self.agg_method not in ["mean", "sum", "min", "max", "mode"]:
            raise InvalidParameterError(
                "agg_method",
                actual=self.agg_method,
                reason=(
                    "when 'n_iter > 1', must be one of 'mean', 'sum', 'min', 'max', "
                    "and 'mode'"
                ),
            )

        # Load the keyword arguments for workers
        self._worker_kwargs: list[dict]
        if isinstance(self.workers, int):
            if self.workers < 1:
                raise InvalidParameterError(
                    "workers",
                    actual=self.workers,
                    reason="if given as an integer, must be >= 1",
                )
            self._worker_kwargs = [{} for _ in range(self.workers)]
        elif isinstance(self.workers, list):
            if any(not isinstance(worker, dict) for worker in self.workers):
                raise InvalidParameterError(
                    "workers",
                    actual=self.workers,
                    reason="if given as a list, each element must be a dict",
                )
            self._worker_kwargs = self.workers
        else:
            raise InvalidParameterError(
                "workers",
                actual=self.workers,
                reason="must be an integer >= 1 or a list of dict",
            )

    def _get_score(self, data_item: DataItemType, **kwargs) -> Real | dict[Any, Real]:
        """Evaluate a data item and obtain its score(s).

        :meta public:

        .. note::
            This method is not implemented and must be overridden in subclasses.

        Parameters
        ----------
        data_item : DataItemType
            The data item.
        kwargs
            The additional keyword arguments.

        Returns
        -------
        scores : real or dict
            The evaluated scores, either a single score or a dictionary of subject-
            score pairs.

        Notes
        -----
        Note that if there are mutliple scores obtained, i.e., returning a dictionary,
        remember that the "i" key cannot be included since it is reserved for indexing.

        Moreover, it is recommended *not* to catch the exceptions that cause the
        processing of a data item to fail, since otherwise
        :meth:`BaseEvaluator.evaluate` will not realize that the data item errors out.
        """
        raise NotImplementedError

    async def _aget_score(
        self, data_item: DataItemType, **kwargs
    ) -> Real | dict[Any, Real]:
        """Evaluate a data item and obtain its score(s).

        :meta public:

        This should be the asynchronous version of :meth:`BaseEvaluator._get_score`.
        See :meth:`BaseEvaluator._get_score` for details. If the obtaining process of
        the scores does not involving anything asynchrounous, this can simply be
        overridden as follows:

        .. code-block:: python

            async def _aget_score(self, data_item, **kwargs):
                return self._get_score(data_item, **kwargs)


        However, note that this will cause only one worker to be actually doing all
        the tasks because it will not give away the control of the event loop.

        .. note::
            This method is not implemented and must be overridden in subclasses.
            Moreover, this method must be defined as asynchronous.
        """
        raise NotImplementedError

    def _yield_dataset(
        self, overwrite: bool = False
    ) -> Generator[tuple[int, int, DataItemType], Any, None]:
        """Yield the indices and data items to be done.

        Parameters
        ----------
        overwrite : bool, default=False
            If ``True``, all existing indices will be ignored.

        Yield
        -----
        i : int
            The index of the data item.
        it : int
            The iteration id of the data item, should be range(0, self.n_iter).
        data_item : DataItemType
            The data item.
        """
        df: pd.DataFrame
        existing_indices: list = []
        if not os.path.exists(self.save_path):
            ensure_path(self.save_path)
            df = pd.DataFrame(columns=["i"], dtype=pd.Int64Dtype)
        else:
            df = pd.read_csv(self.save_path)
            if all(subject in df.columns for subject in self.subjects):
                if not overwrite:
                    existing_indices = list(
                        df[
                            ~df[list(self.subjects)]  # type: ignore[type-var]
                            .isna()
                            .any(axis=1)
                        ].loc[:, "i"]
                    )
        self._result_df = df.set_index("i")

        # Yield the indices and corresponding data items
        if self.fmt == "jsonl":
            with open(self.dataset, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i not in existing_indices:
                        for it in range(self.n_iter):
                            yield i, it, json.loads(line)
        elif self.fmt == "json":
            with open(self.dataset, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            for i, item in enumerate(all_data):
                if i not in existing_indices:
                    for it in range(self.n_iter):
                        yield i, it, item
        else:  # self.fmt == "csv"
            all_data = pd.read_csv(self.dataset)
            for i, row in all_data.iterrows():
                if i not in existing_indices:
                    for it in range(self.n_iter):
                        yield i, it, row

    def _check_scores(self, scores: Real | dict[Any, Real]) -> dict[Any, Real]:
        """Check and format the scores.

        Parameters
        ----------
        scores : real or dict
            The evaluation scores of a data item, either a single real number of a
            dictionary of subject-score pairs.

        Returns
        -------
        eval_scores : dict
            If ``scores`` is a single real number, this returns ``{"scores": scores}``.
            Otherwise, this returns ``scores`` itself.
        """
        # mypy not working with numbers.Real
        if isinstance(scores, Real) and not pd.isna(
            scores
        ):  # type: ignore[call-overload]
            if len(self.subjects) == 1:
                return {self.subjects[0]: scores}
            raise ScoringError("Got a single score but multiple subjects are assigned.")
        elif isinstance(scores, dict):
            # mypy not working with numbers.Real
            bad_item = next(
                (
                    (subject, score)
                    for subject, score in scores.items()
                    if (
                        not isinstance(score, Real)
                        or pd.isna(score)  # type: ignore[call-overload]
                    )
                ),
                None,
            )
            if bad_item is not None:
                raise ScoringError(
                    "The scores must be either a real number or a dictionary with "
                    f"real values; got a dictionary but there exists "
                    f"'{bad_item[0]}: {bad_item[1]}' of type '{type(bad_item[1])}'."
                )
            else:
                missing_keys = set(self.subjects).difference(set(scores))
                if missing_keys:
                    missing_keys_repr = ", ".join(map(repr, missing_keys))
                    raise ScoringError(f"Missing keys: {missing_keys_repr}.")
                return {
                    subject: score
                    for subject, score in scores.items()
                    if subject in self.subjects
                }
        else:
            raise ScoringError(
                "The scores must be either a real number or a dictionary with real "
                f"values; got '{scores}' of type '{type(scores)}' instead."
            )

    def evaluate(self, *, overwrite: bool = False) -> bool:
        """Evaluate the specified dataset.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the data in ``save_path``. If ``False``, the
            evaluation will be built upon existing data in ``save_path``, otherwise
            all data will be evaluated are existing data will be overwritten.

        Returns
        -------
        completed : bool
            Whether the task has been completed.
        """
        mlog_path = manage_timed_logs(type(self).__name__)

        def process_func(
            item: tuple[int, int, DataItemType], **kwargs
        ) -> (
            tuple[tuple[int, int, dict[Any, Real]], list[tuple[Any, Any]], None]
            | tuple[None, None, tuple[Any, Any]]
        ):
            """The sequential processing function."""
            i, it, data_item = item
            prefix = f"{i}" if self.n_iter == 1 else f"{i}.{it}"
            eval_scores: dict[Any, Real] | None = None
            norm_msg: list[tuple[Any, Any]] | None = None
            err: Exception | None = None
            err_trace: str | None = None

            # Handle all exceptions
            try:
                scores = self._get_score(data_item, **kwargs)
                eval_scores = self._check_scores(scores)
                norm_msg = [
                    (f"[{prefix}/ IT]", format_data_item(data_item)),
                    (f"[{prefix}/ SC]", eval_scores),
                ]
            except Exception as e:
                err, err_trace = e, traceback.format_exc()

            # Write the log on demand
            if (
                self.logging_mode == "failed"
                and eval_scores is None
                or self.logging_mode == "all"
            ):
                mlog_item = {
                    "time": str(datetime.now()),
                    "index": i,
                    "iter": it,
                    "eval_scores": eval_scores,
                    "norm_msg": str(norm_msg),
                    "err_msg": err_trace,
                }
                with open(mlog_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(mlog_item, ensure_ascii=False) + "\n")

            # Return the information based on success or failure
            if eval_scores is not None and norm_msg is not None:
                return (i, it, eval_scores), norm_msg, None
            return None, None, (f"[{prefix}/ ERR]", f"{type(err).__name__}: {err}")

        async def process_afunc(
            item: tuple[int, int, DataItemType],
            addtlks: list[asyncio.Lock] | None = None,
            **kwargs,
        ) -> tuple[
            tuple[int, int, dict[Any, Real]], list[tuple[Any, Any]], None
        ] | tuple[None, None, tuple[Any, Any]]:
            """The asynchronous processing function."""
            i, it, data_item = item
            prefix = f"{i}" if self.n_iter == 1 else f"{i}.{it}"
            eval_scores: dict[Any, Real] | None = None
            norm_msg: list[tuple[Any, Any]] | None = None
            err: Exception | None = None
            err_trace: str | None = None

            # Handle all exceptions
            try:
                scores = await self._aget_score(data_item, **kwargs)
                eval_scores = self._check_scores(scores)
                norm_msg = [
                    (f"[{prefix}/ IT]", format_data_item(data_item)),
                    (f"[{prefix}/ SC]", eval_scores),
                ]
            except Exception as e:
                err, err_trace = e, traceback.format_exc()

            # Write the log on demand
            if (
                self.logging_mode == "failed"
                and eval_scores is None
                or self.logging_mode == "all"
            ):
                mlog_item = {
                    "time": str(datetime.now()),
                    "index": i,
                    "iter": it,
                    "eval_scores": eval_scores,
                    "norm_msg": str(norm_msg),
                    "err_msg": err_trace,
                }
                assert (
                    isinstance(addtlks, list) and len(addtlks) >= 1
                ), "no lock for log in async mode; this is most likely an ml3m bug."
                async with addtlks[0]:
                    with open(mlog_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(mlog_item, ensure_ascii=False) + "\n")

            # Return the information based on success or failure
            if eval_scores is not None and norm_msg is not None:
                return (i, it, eval_scores), norm_msg, None
            return None, None, (f"[{prefix}/ ERR]", f"{type(err).__name__}: {err}")

        # Activate the asynchronous runner (sequential mode if only one worker)
        runner = AsyncRunner(
            process_func=process_func,
            process_afunc=process_afunc,
            verbose=self.verbose,
        )
        results: list[tuple[int, int, dict[Any, Real]]]
        failed: list[tuple[int, int, DataItemType]]
        results, failed = runner.run(
            items=self._yield_dataset(overwrite=overwrite),
            worker_kwargs=self._worker_kwargs,
            n_locks=1,
        )
        completed = len(failed) == 0

        # Aggregate the result if needed
        new_df: pd.DataFrame
        failed_indices = [item[0] for item in failed]
        if self.n_iter == 1:
            i2scores = {i: scores for i, _, scores in results}
            new_df = pd.DataFrame(i2scores).T
        else:
            iit2scores = {(i, it): scores for i, it, scores in results}
            grp = pd.DataFrame(iit2scores).T.groupby(level=0)
            if self.agg_method in ["mean", "sum", "min", "max"]:
                new_df = grp.agg(self.agg_method)
            else:  # self.agg_method == "mode":
                # TODO: maybe resolve: error: "Iterable[Any]" has no attribute "mode"
                new_df = grp.apply(
                    lambda x: x.mode().iloc[0]  # type: ignore[attr-defined]
                )

        # Update the file with the obtained results
        for i, row in new_df.iterrows():
            # TODO: maybe resolve: error: Invalid index type "tuple[Hashable | None,
            # Index]" for "_LocIndexerFrame"; expected type ...
            # not sure whether the current operation is perfectly valid in all cases
            self._result_df.loc[i, self.subjects] = row  # type: ignore[index]
        for i in failed_indices:
            self._result_df.loc[i, self.subjects] = None
        self._result_df = self._result_df.convert_dtypes().sort_index()
        self._result_df.reset_index(names="i").to_csv(self.save_path, index=False)

        # Summarize the save location (and possibly log location)
        print(colored("Results can be found at:", COLOR.GREEN))
        print(os.path.abspath(self.save_path))
        if self.logging_mode != "none" and os.path.exists(mlog_path):
            print(colored("Execution log can be found at:", COLOR.GREEN))
            print(os.path.abspath(os.path.abspath(mlog_path)))
        return completed

    def load_scores(
        self,
        subject_subset: list | None = None,
        items: list | None = None,
    ) -> pd.DataFrame:
        """Load the scores from the save location.

        Parameters
        ----------
        subject_subset : list or None
            The subjects of the scores to select, i.e., the columns. If ``None``,
            select all subjects. In the returned :class:`pd.DataFrame`, the columns
            will be in the same order as ``subject_subset``.
        items : list or None
            The indices of the items to select. If ``None``, select all items. This
            will be applied after ``subject_subset``. This does not necessarily need
            to be a subset of the index of the loaded :class:`pd.DataFrame`. The
            indices that do not exist in the index of the loaded :class:`pd.DataFrame`
            will be assigned ``NaN``. In the returned :class:`pd.DataFrame`, the rows
            will be in the same order as ``items``.

        Returns
        -------
        scores : pandas.DataFrame
            The scores loaded from ``save_path``.

        Examples
        --------
        Suppose that the file at ``save_path`` looks like the following:

        .. code-block::

            i,score1,score2
            0,78,83
            1,64,76
            3,100,92
            4,28,38
            5,30,45

        >>> evaluator.load_scores()  # doctest: +SKIP
           score1  score2
        i
        0      78      83
        1      64      76
        3     100      92
        4      28      38
        5      30      45

        >>> evaluator.load_scores(subject_subset=["score2"])  # doctest: +SKIP
           score2
        i
        0      83
        1      76
        3      92
        4      38
        5      45

        >>> evaluator.load_scores(items=list(range(7)))  # doctest: +SKIP
           score1  score2
        i
        0    78.0    83.0
        1    64.0    76.0
        2     NaN     NaN
        3   100.0    92.0
        4    28.0    38.0
        5    30.0    45.0
        6     NaN     NaN
        """
        result_df = pd.read_csv(self.save_path).set_index("i")
        if subject_subset is not None:
            result_df = result_df[subject_subset]
        if items is not None:
            result_df = result_df.reindex(items)
        return result_df

    def load_avg_score(
        self,
        subject_subset: list | None = None,
        items: list | None = None,
    ) -> dict[str, Real]:
        """Load the average score of each subject from the save location.

        Parameters
        ----------
        subject_subset : list or None
            The subjects of the scores to select, i.e., the columns. If ``None``,
            select all subjects.
        items : list or None
            The indices of the items to select. If ``None``, select all items. This
            will be applied after ``subject_subset``. This does not necessarily need
            to be a subset of the index of the loaded :class:`pd.DataFrame`. However,
            any item out of range would not be taken into account when computing the
            average score.

        Returns
        -------
        avg_score : pandas.DataFrame
            The average score loaded from ``save_path``.

        Examples
        --------
        Suppose that the file at ``save_path`` looks like the following:

        .. code-block::

            i,score1,score2
            0,78,83
            1,64,76
            2,100,92
            3,28,38
            4,30,45

        >>> evaluator.load_avg_score()  # doctest: +SKIP
        {'score1': 60.0, 'score2': 66.8}

        >>> evaluator.load_avg_score(subject_subset=["score2"])  # doctest: +SKIP
        {'score2': 66.8}

        >>> evaluator.load_avg_score(items=list(range(7)))  # doctest: +SKIP
        {'score1': 60.0, 'score2': 66.8}
        """
        result_df = self.load_scores(subject_subset=subject_subset, items=items)
        return result_df.mean().to_dict()


class BaseOpenAIEvaluator(BaseEvaluator):
    """Base evaluator class via OpenAI.

    .. note::
        This class is meant to be subclassed. The methods that must be overriden
        include:

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
        ``overwrite`` when using the :meth:`BaseOpenAIEvaluator.evaluate` method.
    subjects : list
        The subjects to evaluate. This should *strictly* correspond to how to scores
        are obtained in :meth:`BaseOpenAIEvaluator._extract_scores`. If the score is
        obtained as a single real value, ``subjects`` must be a list of one element and
        that element will be used as the name of that score. If the score(s) are
        obtained as a dictionary of subject-value pairs, all items in ``subjects`` must
        appear as the keys of the obtained dictionary. Any additional key will be
        discarded; any missing key will be treated as an error.
    openai_config : str or pathlib.Path
        The absolute path to the OpenAI configuration file.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    n_iter : int, default=1
        The number of iterations for each data item. This is commonly used when one
        round of scoring is not convincing enough and the final scoring should be some
        statistics of multiple rounds of scoring.
    agg_method : {"mean", "sum", "min", "max", "mode"}, default=None
        The aggregate method to use on multiple rounds of scoring. Ignored when
        ``n_iter=1``, and otherwise this must not be ``None``.
    timeout : float, default=60
        The timeout in seconds. This is not the OpenAI timeout, but the timeout for
        cancelling the worker tasks.
    model : str, default="gpt-3.5-turbo"
        The ID of the model to use, must be one of the available OpenAI models that
        support the ChatCompletion API. See also
        https://platform.openai.com/docs/models/model-endpoint-compatibility.
    logging_mode : {"all", "failed", "none"}, default="all"
        The logging mode, whether to save the logs of all items, or only of failed
        items, or save no log.
    verbose : int, default=0
        The verbosity level of the processing. For negative levels, only a progress bar
        will be displayed. For level 0, the errored items will also be displayed. For
        positive levels, the all items will be displayed, and the verbosity level
        determines the number of lines to display for the message of each item.
    openai_kwargs
        The additional keyword arguments to pass to OpenAI ChatCompletion. See the
        arguments marked as *Optional* at
        https://platform.openai.com/docs/api-reference/chat/create. Do not try to pass
        ``api_key`` and ``api_base`` through ``openai_kwargs``, use a configuration
        file instead.
    """

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        subjects: list,
        openai_config: str | Path,
        *,
        fmt: DatasetFormat = "jsonl",
        n_iter: int = 1,
        agg_method: AggregateMethod | None = None,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        logging_mode: LoggingMode = "all",
        verbose: int = 0,
        **openai_kwargs,
    ) -> None:
        self.openai_config = openai_config
        self.timeout = timeout
        self.model = model
        self.openai_kwargs = openai_kwargs

        # Load the OpenAI configurations
        validate_path(self.openai_config)
        worker_kwargs = [
            {"api_key": config.key, "api_base": config.base}
            for config in get_openai_config(self.openai_config)
            for _ in range(config.n_workers)
        ]
        if "api_key" in self.openai_kwargs or "api_base" in self.openai_kwargs:
            raise InvalidParameterError(
                "**openai_kwargs",
                actual=self.openai_kwargs,
                reason=(
                    "'api_key' and 'api_base' should be specified with a "
                    "configuration file"
                ),
            )

        # Inherit from parent
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            subjects=subjects,
            fmt=fmt,
            workers=worker_kwargs,
            n_iter=n_iter,
            agg_method=agg_method,
            logging_mode=logging_mode,
            verbose=verbose,
        )

    def _prompt(self, data_item: DataItemType) -> tuple[str, str]:
        """Return the prompt for evaluation.

        :meta public:

        .. note::
            This method is not implemented and must be overridden in subclasses.

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
        """
        raise NotImplementedError

    def _extract_scores(
        self, reply: str, data_item: DataItemType
    ) -> Real | dict[Any, Real]:
        """Extract the score(s) from the OpenAI model reply (and the data item).

        :meta public:

        .. note::
            This method is not implemented and must be overridden in subclasses.

        Parameters
        ----------
        reply : str
            The OpenAI model reply, from which the score(s) will be extracted.
        data_item : DataItemType
            The data item. This may not be used, but in case the OpenAI model reply
            requires comparion with the data item to give the final score, the data
            item is passed in as well.

        Returns
        -------
        scores : real or dict
            The extracted scores, either a single score or a dictionary of subject-
            score pairs.

        Notes
        -----
        This method should correspond to the :meth:`BaseOpenAIEvaluator._prompt`
        method, in the sense that the formatted evaluation prompt is expected to invoke
        an *extractable* model reply, and this method should extract the score(s) from
        that reply. It can extract either a single score or a dictionary of subject-
        score pairs.

        Note that if there are mutliple scores obtained, i.e., returning a dictionary,
        remember that the "i" key cannot be included since it is reserved for indexing.

        It is recommended *not* to catch the exceptions that cause the extraction of
        scores to fail, since otherwise :meth:`BaseOpenAIEvaluator.evaluate` will not
        realize that the data item errors out.
        """
        raise NotImplementedError

    def _prior_scoring(self, data_item: DataItemType):
        """Get messages prior to OpenAI querying.

        Parameters
        ----------
        data_item : DataItemType
            The data item.

        Returns
        -------
        messages : list of dict
            The messages for OpenAI querying.
        """
        sys_msg, eval_prompt = self._prompt(data_item)
        return (
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": eval_prompt},
            ]
            if sys_msg
            else [{"role": "user", "content": eval_prompt}]
        )

    def _post_scoring(
        self, completion: Any, data_item: DataItemType
    ) -> Real | dict[Any, Real]:
        """Process the OpenAI response posterior to querying.

        Parameters
        ----------
        completion : an OpenAI completion
            The returned OpenAI completion.

        Returns
        -------
        scores : real or dict
            The extracted scores, either a single score or a dictionary of subject-
            score pairs.
        """
        finish_reason = completion["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            raise ValueError(f"Model terminated by '{finish_reason}'")
        return self._extract_scores(
            completion["choices"][0]["message"]["content"], data_item
        )

    def _get_score(self, data_item: DataItemType, **kwargs) -> Real | dict[Any, Real]:
        """:meta private:"""
        messages = self._prior_scoring(data_item)
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            api_key=kwargs["api_key"],
            api_base=kwargs["api_base"],
            **self.openai_kwargs,
        )
        return self._post_scoring(completion, data_item)

    async def _aget_score(
        self, data_item: DataItemType, **kwargs
    ) -> Real | dict[Any, Real]:
        """:meta private:"""
        messages = self._prior_scoring(data_item)
        completion = await asyncio.wait_for(
            openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                api_key=kwargs["api_key"],
                api_base=kwargs["api_base"],
                **self.openai_kwargs,
            ),
            timeout=self.timeout,
        )
        return self._post_scoring(completion, data_item)
