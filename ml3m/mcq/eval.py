import re
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Literal

from ..base.eval import BaseEvaluator, BaseOpenAIEvaluator
from .._docstring import format_docstring
from .._typing import DataItemType, DatasetFormat


_DOCSTRINGS = {}

_DOCSTRINGS[
    "prereq"
] = r"""
    To use this evaluator, a dataset of a set of inputs, expected responses, and actual
    responses need to be prepared. The dataset can take the following formats:

    - ``.jsonl``: Each line will be read by :func:`json.loads`, either as a list or as
      a dictionary. This is to be handled by ``info``.

    - ``.json``: The whole file will be read by :func:`json.load`, and it should be a
      json array (and read as a list). Each item in the list will be treated in the
      same way as each line in the ``.jsonl`` format. It can be either a list or a
      dictionary, to be handled by ``info``.

    - ``.csv``: The whole file will be read by :func:`pandas.read_csv` as a
      :class:`pandas.DataFrame`. Each row is to be handled by ``info``.
"""


@format_docstring(prereq=_DOCSTRINGS["prereq"])
class McqOpenAIEvaluator(BaseOpenAIEvaluator):
    """Evaluator for multiple-choice questions via OpenAI.

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
    info : Callable
        The function that extracts the question, actual answer, and expected answer of
        a data item (specifically, a multiple-choice question). The input parameter
        should be a :class:`pandas.Series`, a list, or a dictionary, depending on
        ``format``. The output should be a tuple of three strings, respectively
        the question, the actual answer to that question, and the expected answer of
        that question. See the notes for examples.
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

    Notes
    -----
    Here are some examples of ``info``:

    Assume that ``dataset`` is in ``.jsonl`` format and each line is of the following
    form: ``{{"instruction": "xxx", "input": "xxx", "output": "xxx", "history": [],
    "response": "xxx"}}``. Then ``info`` can be defined as follows:

    .. code-block:: python

        def info(data_item: dict) -> tuple[str, str, str]:
            question = data_item["instruction"] + "\\n" + data_item["input"]
            actual = data_item["response"]
            expected = data_item["output"]
            return question, actual, expected

    Now assume that ``dataset`` is in ``.csv`` format with columns "question", "A",
    "B", "C", "D", "answer", and "response". Then ``info`` can be defined as follows:

    .. code-block:: python

        def info(data_item: pandas.Series) -> tuple[str, str, str]:
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
        openai_config: str | Path,
        save_path: str | Path,
        info: Callable[[DataItemType], tuple[str, str, str]],
        *,
        format: Literal["jsonl", "json", "csv"] = "jsonl",
        n_iter: int = 3,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        verbose: int = 1,
        err_verbose: int = 1,
    ) -> None:
        self.info = info
        super().__init__(
            dataset=dataset,
            openai_config=openai_config,
            save_path=save_path,
            format=format,
            n_iter=n_iter,
            timeout=timeout,
            model=model,
            verbose=verbose,
            err_verbose=err_verbose,
        )

    def _prompt(self, data_item: DataItemType) -> tuple[str, str]:
        """:meta private:"""
        question, actual, expected = self.info(data_item)
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
                f"answer); got '{stripped_reply}' instead."
            )

    def evaluate(
        self, *, overwrite: bool = False, skip_openai_api_cfm: bool = False
    ) -> None:
        super().evaluate(overwrite=overwrite, skip_openai_api_cfm=skip_openai_api_cfm)

    def sync(self) -> None:
        super().sync()


@format_docstring(prereq=_DOCSTRINGS["prereq"])
class McqRegexEvaluator(BaseEvaluator):
    """Evaluator for multiple-choice questions via regular expression matching.

    {prereq}

    In addition, the expected response *must* be a string of labels, e.g. "C", "AD",
    etc. This is for the sake of convenient regex matching. If this is not possible,
    one should use :class:`McqOpenAIEvaluator` instead.

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the ``evaluate`` method.
    info : Callable
        The function that extracts actual answer, and expected answer of a data item
        (specifically, a multiple-choice question). The input parameter should be a
        :class:`pandas.Series`, a list, or a dictionary, depending on ``format``. The
        output should be a tuple of two strings, respectively the actual answer to that
        question, and the expected answer of that question. See the notes for examples.
    regex_patterns : list of re.Pattern or list of str
        The regular expression patterns to use for extracting the selected options from
        the model responses. These will be matched with :func:`re.findall`. If any flag
        such as :obj:`re.DOTALL` is needed, please use :func:`re.compile` to compile it
        in advance. Some reference patterns can be obtained from
        :func:`mcq_regex_patterns`.
    format : {{"jsonl", "json", "csv"}}, default="jsonl"
        The format of ``dataset``, as specified above.

    Notes
    -----
    Here are some examples of ``info``:

    Assume that ``dataset`` is in ``.jsonl`` format and each line is of the following
    form: ``{{"instruction": "xxx", "input": "xxx", "output": "xxx", "history": [],
    "response": "xxx"}}``. Then ``info`` can be defined as follows:

    .. code-block:: python

        def info(data_item: dict) -> tuple[str, str]:
            actual = data_item["response"]
            expected = data_item["output"]
            return actual, expected

    Now assume that ``dataset`` is in ``.csv`` format with columns "question", "A",
    "B", "C", "D", "answer", and "response". Then ``info`` can be defined as follows:

    .. code-block:: python

        def info(data_item: pandas.Series) -> tuple[str, str, str]:
            expected, actual = data_item[["answer", "response"]]
            return actual, expected
    """

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        info: Callable[[DataItemType], tuple[str, str]],
        regex_patterns: list[re.Pattern | str],
        *,
        format: DatasetFormat = "jsonl",
    ) -> None:
        self.info = info
        self.regex_patterns = regex_patterns
        super().__init__(dataset, save_path, format=format)

    def _get_score(self, data_item: DataItemType) -> Real | dict[Any, Real]:
        """:meta private:"""
        actual, expected = self.info()
        re.findall

    def evaluate(self, *, overwrite: bool = False) -> None:
        super().evaluate(overwrite=overwrite)

    def sync(self) -> None:
        super().sync()
