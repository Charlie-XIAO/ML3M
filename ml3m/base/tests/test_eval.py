import json
import os
import random

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ml3m.base import BaseEvaluator

random.seed(2023)


@pytest.fixture
def load_dataset():
    """Load a dataset."""
    return [
        {
            "instruction": "What is the capital of China?",
            "input": "",
            "output": "The capital of China is Beijing.",
            "response": "Beijing.",
        },
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris.",
            "response": "Paris.",
        },
    ]


def make_dataset(fmt, storage, load_dataset):
    """Make a dataset and return its path."""
    data = load_dataset
    orig_dataset = os.path.join(storage, f"make_dataset__dataset.{fmt}")
    if fmt == "jsonl":
        with open(orig_dataset, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif fmt == "json":
        with open(orig_dataset, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:  # fmt == "csv"
        df = pd.DataFrame(data)
        df.to_csv(orig_dataset, index=False)
    return orig_dataset


class NormalBaseEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset,
        save_path,
        subjects,
        *,
        fmt="jsonl",
        workers=1,
        n_iter=1,
        agg_method=None,
        logging_mode="all",
        verbose=1,
        mode="random",
    ):
        super().__init__(
            dataset,
            save_path,
            subjects,
            fmt=fmt,
            workers=workers,
            n_iter=n_iter,
            agg_method=agg_method,
            logging_mode=logging_mode,
            verbose=verbose,
        )
        self.mode = mode

    async def _aget_score(self, data_item, **kwargs):
        if self.mode == "random" or self.mode.startswith("err_on_"):
            if (
                self.mode.startswith("err_on_instruction_")
                and data_item["instruction"] == self.mode[19:]
            ):
                raise ValueError
            print(self.mode)
            if len(self.subjects) == 1:
                return random.randint(0, 100)
            return {subject: random.randint(0, 100) for subject in self.subjects}
        elif self.mode == "all_err":
            raise ValueError


class TestBaseEvaluator:
    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize("subjects", [["score1"], ["score2", "score3"]])
    @pytest.mark.parametrize("workers", [1, 3])
    @pytest.mark.parametrize("n_iter,agg_method", [(1, None), (3, "sum"), (3, "mode")])
    def test_base_evaluator_result_versus_written(
        self, fmt, subjects, workers, n_iter, agg_method, storage, load_dataset, request
    ):
        """Test that evaluator._result_df and the written csv are the same."""
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        evaluator = NormalBaseEvaluator(
            dataset=make_dataset(fmt, storage, load_dataset),
            save_path=save_path,
            subjects=subjects,
            fmt=fmt,
            workers=workers,
            n_iter=n_iter,
            agg_method=agg_method,
        )
        completed = evaluator.evaluate()
        assert completed

        df_saved = pd.read_csv(save_path)
        df_stored = evaluator._result_df.reset_index(names="i")
        assert_frame_equal(df_saved, df_stored, check_dtype=False)

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize("subjects", [["score1"], ["score2", "score3"]])
    @pytest.mark.parametrize("new_subjects", [["score4"], ["score5", "score6"]])
    @pytest.mark.parametrize("workers", [1, 3])
    def test_base_evaluator_evaluate_basics(
        self, fmt, subjects, new_subjects, workers, storage, load_dataset, request
    ):
        """Test the basic evaluator functionalities.

        Fail all data items
        -> Pass one of the data items
        -> Pass all data items
        -> Evaluate again (should make no change)
        -> Fail all data items on the new subjects
        -> Pass one of the data items on the new subjects
        -> Evaluate on old subject(s) again (should make no change)
        -> Pass all data items on the new subjects
        -> Evaluate again (should make no change)
        """
        data = load_dataset
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        # This should pass none of the data items
        evaluator = NormalBaseEvaluator(
            dataset=make_dataset(fmt, storage, load_dataset),
            save_path=save_path,
            subjects=subjects,
            fmt=fmt,
            workers=workers,
            mode="all_err",
        )
        completed = evaluator.evaluate()
        assert not completed

        df = evaluator._result_df
        assert len(df) == 0
        assert len(df.columns) == 0

        # This should pass the second data item but fail the first
        item_0_instruction = data[0]["instruction"]
        evaluator.mode = f"err_on_instruction_{item_0_instruction}"
        completed = evaluator.evaluate()
        assert not completed

        df = evaluator._result_df
        item_1_scores = df.loc[1, :]
        assert list(df.index) == [1]
        assert list(df.columns) == subjects

        # This should pass all of the data items
        evaluator.mode = "random"
        completed = evaluator.evaluate()
        assert completed

        df = evaluator._result_df
        items_scores = df.copy()
        assert list(df.index) == [0, 1]
        assert list(df.columns) == subjects
        assert not df.isna().any().any()
        assert_series_equal(item_1_scores, df.loc[1, :])

        # This should not modify the evaluation results
        evaluator.evaluate()
        completed = evaluator.evaluate()
        assert completed

        df = evaluator._result_df
        assert_frame_equal(items_scores, df)

        # This should pass none of the data items on the new subjects
        evaluator2 = NormalBaseEvaluator(
            dataset=make_dataset(fmt, storage, load_dataset),
            save_path=save_path,
            subjects=new_subjects,
            fmt=fmt,
            workers=workers,
            mode="all_err",
        )
        completed = evaluator2.evaluate()
        assert not completed

        df = evaluator2._result_df
        assert_frame_equal(items_scores, df)

        # This should pass the second data item but fail the first on the new subjects
        evaluator2.mode = f"err_on_instruction_{item_0_instruction}"
        completed = evaluator2.evaluate()
        assert not completed

        df = evaluator2._result_df
        item_1_scores = df.loc[1, :]
        new_items_scores = df.copy()
        assert pd.isna(df.loc[0, new_subjects]).all()
        assert not pd.isna(df.loc[1, new_subjects]).any()
        assert list(df.columns) == [*subjects, *new_subjects]
        assert_frame_equal(items_scores, df[subjects])

        # This should not modify the evaluation results since it is using old subjects
        completed = evaluator.evaluate()
        assert completed

        df = evaluator._result_df
        assert_frame_equal(new_items_scores, df)

        # This should pass all of the data items on the new subjects
        evaluator2.mode = "random"
        completed = evaluator2.evaluate()
        assert completed

        df = evaluator2._result_df
        new_items_scores = df.copy()
        assert list(df.index) == [0, 1]
        assert list(df.columns) == [*subjects, *new_subjects]
        assert not df.isna().any().any()
        assert_series_equal(item_1_scores, df.loc[1, :])

        # This should not modify the evaluation results
        evaluator2.evaluate()
        completed = evaluator2.evaluate()
        assert completed

        df = evaluator2._result_df
        assert_frame_equal(new_items_scores, df)
