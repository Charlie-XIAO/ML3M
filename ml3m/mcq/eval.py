from numbers import Real
from pathlib import Path
from typing import Any

from ml3m._typing import InputType
from ..utils.eval import BaseOpenAIEvaluator


class McqOpenAIEvaluator(BaseOpenAIEvaluator):
    """Evaluator for multiple-choice questions via OpenAI.

    TODO
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
        super().__init__(
            dataset=dataset,
            openai_config=openai_config,
            save_path=save_path,
            jsonl=jsonl,
            n_iter=n_iter,
            timeout=timeout,
            model=model,
            verbose=verbose,
            err_verbose=err_verbose,
        )

    def prompt(
        self, inputs: InputType, response: dict[str, str], expected: dict[str, str]
    ) -> tuple[str, str]:
        assert len(inputs) == 1  # This class is only for single-round querying
        return (
            "",
            f"### As follows is a multiple-choice question:\n```\n{inputs[0]['value']}"
            f"\n```\n\n### The correct answer to this question is: {expected['value']}"
            f"\n\n### My answer to this question is:\n```\n{response['value']}\n```\n"
            "\nIf my answer is correct, reply '1'. If my answer is incorrect, reply "
            "'0'. Do not include any additional information。"
        )

    def extract_scores(self, reply: str) -> Real | dict[Any, Real]:
        stripped_reply = reply.strip()
        if stripped_reply == "1":
            return 100
        elif stripped_reply == "0":
            return 0
        else:
            raise ValueError(
                "The expected OpenAI response is 0 (incorrect answer) or 1 (correct "
                f"answer); got {stripped_reply} instead."
            )
