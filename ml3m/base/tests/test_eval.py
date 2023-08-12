import asyncio
import json
import os
import random

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ml3m.base import BaseEvaluator
from ml3m.errors import InvalidParameterError

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
    """Extends the base evaluator.

    This can return random scores, fail on all items, or fail on a certain item based
    on ``data_item["instruction"]``.
    """

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


class MultIterBaseEvaluator(BaseEvaluator):
    """Extends the base evaluator.

    This supports taking a function ``sc_mapping`` that maps the iteration index and
    the subject index to a score. It does not tell different data items apart.
    """

    def __init__(
        self,
        dataset,
        save_path,
        subjects,
        sc_mapping,
        *,
        fmt="jsonl",
        workers=1,
        n_iter=1,
        agg_method=None,
        logging_mode="all",
        verbose=1,
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
        self.sc_mapping = sc_mapping
        self.lock = asyncio.Lock()
        self.tracker = {}

    async def _aget_score(self, data_item, **kwargs):
        instruction = data_item["instruction"]
        async with self.lock:
            if instruction in self.tracker:
                ind = self.tracker[instruction]
                self.tracker[instruction] += 1
            else:
                ind = 0
                self.tracker[instruction] = 1
        return {
            subject: self.sc_mapping(ind, i) for i, subject in enumerate(self.subjects)
        }


class TestBaseEvaluator:
    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize("subjects", [["score1"], ["score2", "score3"]])
    @pytest.mark.parametrize("workers", [1, 3])
    @pytest.mark.parametrize("n_iter,agg_method", [(1, None), (3, "sum"), (3, "mode")])
    def test_base_evaluator_result_versus_written(
        self, fmt, subjects, workers, n_iter, agg_method, storage, load_dataset, request
    ):
        """Test that evaluator._result_df and the written csv are the same."""
        dataset = make_dataset(fmt, storage, load_dataset)
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        evaluator = NormalBaseEvaluator(
            dataset=dataset,
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
        dataset = make_dataset(fmt, storage, load_dataset)
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        # This should pass none of the data items
        evaluator = NormalBaseEvaluator(
            dataset=dataset,
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
            dataset=dataset,
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

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize("workers", [1, 3])
    @pytest.mark.parametrize("agg_method", ["mean", "sum", "min", "max", "mode"])
    def test_base_evaluator_aggregate(
        self, fmt, workers, agg_method, storage, load_dataset, request
    ):
        """Test the aggregation methods."""
        subjects = ["score1", "score2"]
        dataset = make_dataset(fmt, storage, load_dataset)
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        evaluator = MultIterBaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=subjects,
            sc_mapping=lambda i_iter, i_subject: i_iter % 4 + i_subject * 10,
            fmt=fmt,
            workers=workers,
            n_iter=5,
            agg_method=agg_method,
        )
        evaluator.evaluate()

        # For each data item, the scores would be
        # score1: 0, 1, 2, 3, 0
        # score2: 10, 11, 12, 13, 10

        if agg_method == "mean":
            # score1: (0 + 1 + 2 + 3 + 0) / 5 = 1.2
            # score2: (10 + 11 + 12 + 13 + 10) / 5 = 11.2
            results = [1.2, 11.2]
        elif agg_method == "sum":
            # score1: 0 + 1 + 2 + 3 + 0 = 6
            # score2: 10 + 11 + 12 + 13 + 10 = 56
            results = [6, 56]
        elif agg_method == "min":
            # score1: min(0, 1, 2, 3, 0) = 0
            # score2: min(10, 11, 12, 13, 10) = 10
            results = [0, 10]
        elif agg_method == "max":
            # score1: max(0, 1, 2, 3, 0) = 3
            # score2: max(10, 11, 12, 13, 10) = 13
            results = [3, 13]
        elif agg_method == "mode":
            # score1: mode(0, 1, 2, 3, 0) = 0
            # score2: mode(10, 11, 12, 13, 10) = 10
            results = [0, 10]

        expected = pd.DataFrame([results] * 2, columns=subjects)
        assert_frame_equal(
            evaluator._result_df, expected, check_dtype=False, check_names=False
        )

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    def test_base_evalutor_overlapping_subjects(
        self, fmt, storage, load_dataset, request
    ):
        """Test the behavior when two evaluations have overlapping subject.

        The overlapped subject will be overwritten without any warning or error.
        """
        dataset = make_dataset(fmt, storage, load_dataset)
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        evaluator = NormalBaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=["score", "score1"],
            fmt=fmt,
        )
        completed = evaluator.evaluate()
        df = evaluator._result_df
        assert completed

        evaluator2 = NormalBaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=["score", "score2"],
            fmt=fmt,
        )
        completed = evaluator2.evaluate()
        df2 = evaluator2._result_df
        assert completed

        assert list(df2.columns) == ["score", "score1", "score2"]
        assert not df2.isna().any().any()

        # score1 will stay the same, but score should be overwritten
        # Since the evaluator is in random mode, most probably the overwritten part
        # will not be the same as what it previously was
        assert_series_equal(df.loc[:, "score1"], df2.loc[:, "score1"])
        with pytest.raises(AssertionError):
            assert_series_equal(df.loc[:, "score"], df2.loc[:, "score"])

    def test_base_evalutor_invalid_init(self, storage, load_dataset, request):
        """Test invalid initialization."""
        dataset = make_dataset("jsonl", storage, load_dataset)
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        # Test invalid subjects
        for subjects in [[], "abc", ["i", "j"]]:
            with pytest.raises(InvalidParameterError):
                BaseEvaluator(
                    dataset=dataset,
                    save_path=save_path,
                    subjects=subjects,
                )

        # Test invalid fmt
        for fmt in [".csv", "txt"]:
            with pytest.raises(InvalidParameterError):
                BaseEvaluator(
                    dataset=dataset,
                    save_path=save_path,
                    subjects=["score"],
                    fmt=fmt,
                )

        # Test invalid logging_mode
        for logging_mode in ["any", "succeeded"]:
            with pytest.raises(InvalidParameterError):
                BaseEvaluator(
                    dataset=dataset,
                    save_path=save_path,
                    subjects=["score"],
                    logging_mode=logging_mode,
                )

        # Test invalid n_iter and agg_method
        for n_iter, agg_method in zip(
            [0, 2.4, "3", 2, 2, 2],
            ["mean", "mean", "mean", None, "cumsum"],
        ):
            with pytest.raises(InvalidParameterError):
                BaseEvaluator(
                    dataset=dataset,
                    save_path=save_path,
                    subjects=["score"],
                    n_iter=n_iter,
                    agg_method=agg_method,
                )

        # Test invalid workers
        for workers in [{"key": 0}, 0, 2.4, [0, 1]]:
            with pytest.raises(InvalidParameterError):
                BaseEvaluator(
                    dataset=dataset,
                    save_path=save_path,
                    subjects=["score"],
                    workers=workers,
                )
