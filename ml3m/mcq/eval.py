from numbers import Real
from pathlib import Path
from typing import Any, Callable

from .._typing import DataItemType, DatasetFormat, LoggingMode
from ..base.eval import BaseOpenAIEvaluator


class McqOpenAIEvaluator(BaseOpenAIEvaluator):
    """Evaluator for multiple-choice questions via OpenAI.

    This evaluator utilizes the ability of OpenAI models to tell if a response selects
    the correct options, based on the reference answer. The score for each data item
    would be either 0 or 100, and there will be no partial credits.

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the :meth:`McqOpenAIEvaluator.evaluate` method.
    openai_config : str or pathlib.Path
        The absolute path to the OpenAI configuration file.
    info_func : Callable
        The function that extracts the question, actual answer, and expected answer of
        a data item (specifically, a multiple-choice question). The input parameter
        should be a :class:`pandas.Series`, a list, or a dictionary, depending on
        ``fmt``. The output should be a tuple of three strings, respectively the
        question, the actual answer to that question, and the expected answer of that
        question. See the notes for examples.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    n_iter : int, default=1
        The number of iterations for each data item. The mode of the scores for each
        data item will be taken as the final score.
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

    Notes
    -----
    Here are some examples of ``info_func``:

    Assume that ``dataset`` is in ``.jsonl`` format and each line is of the following
    form: ``{{"instruction": "xxx", "input": "xxx", "output": "xxx", "history": [],
    "response": "xxx"}}``. Then ``info_func`` can be defined as follows:

    .. code-block:: python

        def info_func(data_item: dict) -> tuple[str, str, str]:
            question = data_item["instruction"] + "\\n" + data_item["input"]
            actual = data_item["response"]
            expected = data_item["output"]
            return question, actual, expected

    Now assume that ``dataset`` is in ``.csv`` format with columns "question", "A",
    "B", "C", "D", "answer", and "response". Then ``info_func`` can be defined as
    follows:

    .. code-block:: python

        def info_func(data_item: pandas.Series) -> tuple[str, str, str]:
            question, A, B, C, D, answer, response = data_item[
                ["question", "A", "B", "C", "D", "answer", "response"]
            ]
            formatted_question = (
                f"{{question}}\\nA. {{A}}\\nB. {{B}}\\nC. {{C}}\\nD. {{D}}"
            )
            return formatted_question, response, answer
    """

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        openai_config: str | Path,
        info_func: Callable[[DataItemType], tuple[str, str, str]],
        *,
        fmt: DatasetFormat = "jsonl",
        n_iter: int = 1,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        logging_mode: LoggingMode = "all",
        verbose: int = 1,
    ) -> None:
        self.info_func = info_func
        if not callable(info_func):
            raise ValueError("info_func must be a callable.")

        # Inherit from parent
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            openai_config=openai_config,
            fmt=fmt,
            n_iter=n_iter,
            agg_method="mode",
            timeout=timeout,
            model=model,
            logging_mode=logging_mode,
            verbose=verbose,
        )

    def _prompt(self, data_item: DataItemType) -> tuple[str, str]:
        """:meta private:"""
        question, actual, expected = self.info_func(data_item)
        return (
            "",
            f"### As follows is a multiple-choice question:\n```\n{question}\n```\n\n"
            f"### The correct answer to this question is: {actual}\n\n### My answer "
            f"to this question is:\n```\n{expected}\n```\n\nIf my answer is correct, "
            "reply '1'. If my answer is incorrect, reply '0'. Do not include any "
            "additional information.",
        )

    def _extract_scores(self, reply: str) -> Real | dict[Any, Real]:
        """:meta private:"""
        stripped_reply = reply.strip()
        if stripped_reply == "1":
            return 100
        elif stripped_reply == "0":
            return 0
        else:
            raise ValueError(
                "The expected OpenAI response is 0 (incorrect answer) or 1 (correct "
                f"answer); got '{reply}' instead."
            )
