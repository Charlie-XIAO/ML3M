from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any, Callable, Literal

from ..base.eval import BaseOpenAIEvaluator
from ..errors import InvalidParameterError

if TYPE_CHECKING:
    from numbers import Real
    from pathlib import Path

    from .._typing import DataItemType, DatasetFormat, LoggingMode


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
        a data item. The input parameter should be a :class:`pandas.Series`, a list, or
        a dictionary, depending on ``fmt`` and the specific type of each data item. The
        output should be a tuple of three strings, respectively the question, the actual
        answer to that question, and the expected answer of that question. See the notes
        for examples.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    score_name : str, default="score"
        The key/column name to use for the obtained score. This should *not* be a key
        or column name that already exists in the save location. Be extremely careful
        since there will be *no* warning or exception raised on this.
    label_type : {"upper", "lower", "digit"}, default="upper"
        The type of the option labels. "upper" stands for A, B, C, D, ... "lower"
        stands for a, b, c, d, ... "digit" stands for 1, 2, 3, 4, ...
    label_cnt : int, default=4
        The number of options. For instance, ``label_type="upper"`` with
        ``label_cnt=4`` means that the option labels are A, B, C, and D.
    setting: str, optional
        The personality setting for the OpenAI model, passed as the system message. If
        ``None``, then no system message is used.
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
    verbose : int, default=0
        The verbosity level of the processing. For negative levels, only a progress bar
        will be displayed. For level 0, the errored items will also be displayed. For
        positive levels, the all items will be displayed, and the verbosity level
        determines the number of lines to display for the message of each item.

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
        score_name: str = "score",
        label_type: Literal["upper", "lower", "digit"] = "upper",
        label_cnt: int = 4,
        setting: str | None = None,
        n_iter: int = 1,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo",
        logging_mode: LoggingMode = "all",
        verbose: int = 0,
    ) -> None:
        self.info_func = info_func
        self.score_name = score_name
        self.label_type = label_type
        self.label_cnt = label_cnt
        self.setting = setting

        # Validate the arguments
        if not callable(self.info_func):
            raise InvalidParameterError(
                "info_func",
                actual=self.info_func,
                reason="must be a callable",
            )
        if label_type not in ["upper", "lower", "digit"]:
            raise InvalidParameterError(
                "label_type",
                actual=self.label_type,
                reason="must be one of 'upper', 'lower', and 'digit'.",
            )
        if not isinstance(label_cnt, int) or label_cnt <= 1:
            raise InvalidParameterError(
                "label_cnt",
                actual=self.label_cnt,
                reason="must be an integer > 1",
            )

        # Determine the actual labels
        self.labels: list[str]
        if label_type == "upper":
            if self.label_cnt > 13:
                raise ValueError("'label_type=upper' supports at most 13 labels.")
            self.labels = [chr(65 + i) for i in range(self.label_cnt)]
        elif label_type == "lower":
            if self.label_cnt > 26:
                raise ValueError("'label_type=lower' supports at most 26 labels.")
            self.labels = [chr(97 + i) for i in range(self.label_cnt)]
        else:  # label_type == "digit"
            self.labels = [str(i + 1) for i in range(self.label_cnt)]

        # Prepare for querying and reply processing
        label_grp = "|".join(self.labels)
        label_spls = [
            "".join(sorted(random.sample(self.labels, i + 1)))
            for i in range(min(3, self.label_cnt))
        ]
        self._few_shot = ", ".join(f"'{spl}'" for spl in label_spls)
        self._pat = re.compile(
            rf"selected option(?:\(s\)|s)? (is|are|is/are):? ([{label_grp}]+)"
        )

        # Inherit from parent
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            subjects=[self.score_name],
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
        question, actual, _ = self.info_func(data_item)
        return (
            "" if self.setting is None else self.setting,
            "In this task, I will provide you a multiple-choice question and a "
            "student's answer to it. I want you to tell which options the student has "
            "selected as the solution to the multiple-choice question. The criteria "
            "are as follows:\n- If the student's answer explicitly mentions which "
            "are correct and which are wrong, he selects the correct ones.\n- If the "
            "student's answer only lists several (or all) options without saying "
            "whether they are correct or wrong, he selects all options he has listed."
            "\n- If none of the above criteria applies to the student's answer, judge "
            "from the content of his answer.\n\nThe multiple-choice question is:\n```"
            f"\n{question}\n```\n\nThe student's answer is:\n```\n{actual}\n```\n\n"
            "Remember that when you output the student's selected options, only "
            f"include the labels in the format e.g. {self._few_shot}. If the student "
            "selects none of the options, output only 'N'. Do not include any "
            "explanation or addidtional information! The student's selected option(s) "
            "is/are:",
        )

    def _extract_scores(
        self, reply: str, data_item: DataItemType
    ) -> Real | dict[Any, Real]:
        """:meta private:"""
        stripped_reply = reply.strip()

        # Try to match the reply pattern in advance if possible
        mat = re.search(self._pat, stripped_reply)
        if mat is not None:
            stripped_reply = mat.group(2)

        # Construct the set of chosen options
        chosen_options: set[str] = set()
        for char in stripped_reply:
            if not char.isspace() and char not in ",.，、" and char not in self.labels:
                raise ValueError(
                    f"Got invalid character '{char}' in '{stripped_reply}'."
                )
            chosen_options.add(char)

        # Compare with the reference answer
        _, _, expected = self.info_func(data_item)
        expected_options: set[str] = set()
        for char in expected:
            if not char.isspace() and char not in ",.，、" and char not in self.labels:
                raise ValueError(
                    f"[FATAL] Got invalid reference answer '{expected}' in the dataset."
                )
            expected_options.add(char)
        # mypy not working with numbers.Real
        return (expected_options == chosen_options) * 100  # type: ignore[return-value]
