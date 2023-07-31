import asyncio
import json
import os
import traceback
import warnings
from datetime import datetime
from functools import partial
from numbers import Real
from pathlib import Path
from typing import Any, Coroutine, Generator, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from .uopenai import _openai_chatcompletion, get_openai_config, OpenAIConfig
from .._typing import InputType


class BaseOpenAIEvaluator:
    """Base evaluator class via OpenAI.

    This class is meant to be subclassed. The methods that must be overriden include:
    - self.prompt(
        inputs: InputType, response: dict[str, str], expected: dict[str, str]
    ) -> tuple[str, str]:
    - self.extract_scores(reply: str) -> Real | dict

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset, including the full reference
        conversations and actual model responses.
    openai_config : str or pathlib.Path
        The absolute path to the OpenAI configuration file.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Therefore, one should not intentionally create an empty file because an empty
        file cannot be loaded as valid DataFrame. Whether to overwrite the existing
        results or to build on them depend on ``overwrite`` when using the ``evaluate``
        method.
    jsonl : bool, default=False
        Whether ``dataset`` and ``responses`` are in jsonl format. If ``False``, they
        should be in json format (a json array). No matter ``True`` or ``False``,
        ``dataset`` and ``responses`` should follow the same format.
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
        jsonl: bool = False,
        n_iter: int = 3,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        verbose: int = 1,
        err_verbose: int = 1,
    ) -> None:
        self.dataset = dataset
        self.openai_config = openai_config
        self.save_path = save_path
        self.jsonl = jsonl
        self.n_iter = n_iter
        self.timeout = timeout
        self.model = model
        self.verbose = verbose
        self.err_verbose = err_verbose

        # Validate the paths
        if not os.path.exists(self.dataset):
            raise ValueError(f"Dataset not found at {os.path.abspath(self.dataset)}")
        if not os.path.exists(self.openai_config):
            raise ValueError(
                "OpenAI configuration not found at "
                f"{os.path.abspath(self.openai_config)}"
            )
        self.sync()

        # Synchronization locks
        self._tqdmlk = asyncio.Lock()  # For tqdm progress bar update
        self._mloglk = asyncio.Lock()  # For writing log of model responses
        self._mainlk = asyncio.Lock()  # For collecting data

    def prompt(
        self, inputs: InputType, response: dict[str, str], expected: dict[str, str]
    ) -> tuple[str, str]:
        """Return the prompt for evaluation.

        This method must be overridden. It should format the input/instruction, the
        actual model response, and the expected/reference model response into a prompt
        for evaluation. The output format in which the OpenAI model should be prompted
        to reply depends on the ``extract_scores`` method, which determines how the
        OpenAI model response is going to be handled. The response is expected to be
        extracted either as a single scores or as a dictionary of subject-score pairs.

        If single-round, ``inputs`` is of the form e.g.

            ```
            [{"from": "human", "value": "xxx"}]
            ```

        If multiple-round, ``inputs`` is of the form e.g.

            ```
            [{"from": "human", "value": "xxx"},
             {"from": "assistant", "value": "xxx"},
             {"from": "human", "value": "xxx"}]
            ```
        
        Parameters
        ----------
        inputs : InputType
            The input of the format specified as above.
        response : dict[str, str]
            The actual model response with keys "from" and "value".
        expected : dict[str, str]
            The expected model response with keys "from" and "value".

        Returns
        -------
        sys_msg : str or None
            The system message for setting the role of the OpenAI model when querying
            for evaluation, e.g. a professional teacher in some field.
        eval_prompt : str
            The formatted prompt. If the evaluation data includes single-round queries,
            then at least the single-round format should be handled. If it includes
            multiple-round queries, then at least the multiple-round format should be
            handled.
        """
        raise NotImplementedError

    def extract_scores(self, reply: str) -> Real | dict[Any, Real]:
        """Extract the scores from the OpenAI model reply.

        This method must be overridden. It should correspond to the ``prompt`` method,
        which formats the evaluation prompt. This should either extract a single score,
        or extract a dictionary of subject-score pairs.

        It is fine that this method raises an exception, because this will be properly
        caught and treated as an error. In other words, if the reply message is not as
        expected and essential information cannot be properly extracted, this method
        should properly raise exceptions.
        
        Parameters
        ----------
        reply : str
            The OpenAI model reply, from which the score(s) will be extracted.

        Returns
        -------
        scores : numbers.Real or dict
            The extracted scores, either a single score or a dictionary of subject-
            score pairs.
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
        if hasattr(self, "_completed") and self._completed:
            print("The evaluation has been fully completed.")
            return
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

        # Activate the main event loop
        base = os.path.join(os.path.dirname(__file__), "..", "..", "openai_model_logs")
        if not os.path.exists(base):
            os.makedirs(base)
        curtime = datetime.now().strftime("%Y-%m-%d-%H.%M.%S.%f")
        mlog_path = os.path.join(base, f"{curtime}_openai_model.log")
        if overwrite:
            self._df = pd.DataFrame(columns=["i"], dtype=np.int64)
        self._yield_data_in_iteration = partial(self._yield_data, overwrite=overwrite)
        for it in range(self.n_iter):
            asyncio.run(self._mainloop(it=it, mlog_path=mlog_path))
            if self.completed(lazy=True):
                break

        # Write the latest updated DataFrame
        print("Updating results...", end=" ", flush=True)
        self._df.convert_dtypes().sort_values(by=["i"]).to_csv(self.save_path, index=False)
        print(f"done, available at:\n{os.path.abspath(self.save_path)}")

    def sync(self) -> None:
        """Sync up with the results in the save path.

        This method should be called whenever the file at ``save_path`` is modified yet
        one still uses the original evaluator instance.
        """
        abs_save_path = os.path.abspath(self.save_path)
        if not os.path.exists(self.save_path):
            warnings.warn(
                f"Save path not found at {abs_save_path}; forcefully created",
                UserWarning,
                stacklevel=2,
            )
            directories, _ = os.path.split(abs_save_path)
            if not os.path.exists(directories):
                os.makedirs(directories)
            self._df = pd.DataFrame(columns=["i"], dtype=np.int64)
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

    def completed(self, lazy=False) -> bool:
        """Determine hether the evalution task has been fully completed.

        Parameters
        ----------
        lazy : bool, default=False
            Whether to use lazy evaluation when possible. Lazy evaluation is possible
            when the method ``evaluate`` has been called at least once, in which case
            the observation of the last iteration when the last time ``evaluate`` is
            called would be directly used. Note that lazy evaluation will not read
            ``dataset`` again, so it should not be used if the contents of ``dataset``
            has been updated.

        Returns
        -------
        completed : bool
            Whether the evaluation task has been fully completed.
        """
        if lazy and hasattr(self, "_completed"):
            return self._completed
        return len(list(self._yield_data())) == 0

    def _yield_data(
        self, overwrite=False
    ) -> Generator[tuple[int, InputType, dict[str, str], dict[str, str]], Any, None]:
        """Yield the indices and data items to be done.

        Yield
        -----
        i : int
            The index of the data item. It is the index in the json array if the
            dataset is in json format or the index of the line if the dataset is in
            jsonl format.
        inputs : InputType
            The input of the format specified as above.
        response : dict[str, str]
            The actual response with keys "from" and "value".
        expected : dict[str, str]
            The desired response with keys "from" and "value".
        """
        existing_indices: list | pd.Series

        # Get the existing indices that will not be yielded
        if overwrite:
            existing_indices = []
        else:
            existing_indices = self._df.loc[:, "i"]

        # If jsonl, read line by line, skipping the existing indices
        if self.jsonl:
            with open(self.dataset, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i not in existing_indices:
                        item = json.loads(line)
                        yield (
                            i,
                            item["conversations"][:-1],
                            item["response"],
                            item["conversations"][-1],
                        )

        # If not jsonl, the whole dataset has to be loaded into memory at once due to
        # json parsing mechanisms; then yield skipping the existing indices
        else:
            with open(self.dataset, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            for i, item in enumerate(all_data):
                if i not in existing_indices:
                    yield (
                        i,
                        item["conversations"][:-1],
                        item["response"],
                        item["conversations"][-1],
                    )

    async def _execute(
        self,
        *,
        queue: asyncio.Queue[tuple[int, InputType, dict[str, str], dict[str, str]]],
        shared_resources: list[
            tuple[int, dict[Any, Real], Literal[True]]
            | tuple[
                int, tuple[InputType, dict[str, str], dict[str, str]], Literal[False]
            ]
        ],
        openai_config: OpenAIConfig,
        mlog_path : str | Path,
        progbar: tqdm,
        it_id: int,
        worker_id: int,
        openai_api_id: int,
    ):
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

        Returns
        -------
        i : int
            The index of the processed data item.
        result : dict or (InputType, str)
            The subject-score pairs. If the ``extract_scores`` method returns a single
            score, then the subject would be named "scores". If any exception occurred
            during the process, or if the scores are not of valid types, this would be
            ``inputs`` and ``expected`` of that data item for further iterations.
        passed : bool
            Whether the subject-score pairs are successfully obtained.
        """
        while True:
            i, inputs, response, expected = await queue.get()
            sys_msg, eval_prompt = self.prompt(inputs, response, expected)

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
                "api_id": openai_api_id,
                "api_key": openai_config.key,
                "reply": reply,
                "usage": usage,
                "errmsg": errmsg,
            }
            if errmsg is None:
                try:
                    scores = self.extract_scores(reply)
                    if isinstance(scores, Real) and not pd.isna(scores):
                        eval_scores = {"score": scores}
                    elif isinstance(scores, dict):
                        bad_item = next(
                            (
                                item for item in scores.items()
                                if not isinstance(item[1], Real) or pd.isna(item[1])
                            ),
                            None,
                        )
                        if bad_item is not None:
                            raise TypeError(
                                f"``{type(self).__name__}.extract_scores`` must "
                                f"return a dict of Real or a Real, got ``{scores}`` "
                                f"of type dict but there exists {bad_item[0]}: "
                                f"{bad_item[1]} of type ``{type(bad_item[1])}."
                            )
                        else:
                            eval_scores = scores
                    else:
                        raise TypeError(
                            f"``{type(self).__name__}.extract_scores`` must return a "
                            f"dict of Real or a Real, got ``{scores}`` of type "
                            f"{type(scores)} instead."
                        )
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
                        i, (inputs, response, expected), False
                    )
                )
            queue.task_done()

    async def _mainloop(
        self, *, it: int, mlog_path: str | Path
    ) -> Coroutine[Any, Any, None]:
        """Main event loop for asynchronous querying.

        Parameters
        ----------
        it : int
            The id of the current iteration.
        mlog_path : str or pathlib.Path
            The path to the model response log.
        """
        queue: asyncio.Queue[tuple[int, InputType, str]] = asyncio.Queue()
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
            | tuple[
                int, tuple[InputType, dict[str, str], dict[str, str]], Literal[False]
            ]
        ] = []
        openai_configs = get_openai_config(self.openai_config)
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
        todo_items: list[tuple[int, InputType, dict[str, str], dict[str, str]]] = []
        for i, result, passed in shared_resources:
            if passed:
                result_scores[i] = result
            else:
                todo_items.append((i, *result))
        if todo_items:
            print(f"\033[31m{len(todo_items)} failed\033[0m among all {n_items} items")
        else:
            self._completed = True
            print(f"\033[92mall {n_items} items passed\033[0m")

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
