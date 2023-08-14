import asyncio
import json
import os
import random
import re
import threading
from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ml3m.base import BaseEvaluator, BaseOpenAIEvaluator
from ml3m.errors import InvalidParameterError, ScoringError

random.seed(2023)


#######################################################################################
#                                                                                     #
#                                        DATA                                         #
#                                                                                     #
#######################################################################################


dataset_2 = [
    {
        "instruction": "What is the capital of China?",
        "input": "",
        "output": "The capital of China is Beijing.",
        "response": "Beijing.",  # This is clearly correct
    },
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris.",
        "response": "Marseille.",  # And this is clearly incorrect
    },
]

result_df_5_2 = pd.DataFrame(
    data=np.array([[78, 83], [64, 76], [100, 92], [28, 38], [30, 45]]),
    index=pd.Index([0, 1, 3, 4, 5], name="i"),
    columns=["score1", "score2"],
)

openai_configuration = [{"key": "sk-xxx", "base": None, "n_workers": 30}]

openai_configuration_1w = [{"key": "sk-xxx", "base": None, "n_workers": 1}]


@pytest.fixture(scope="module")
def prepare(request, storage):
    """Make a temporary storage and clear it towards the end."""
    paths = {}

    # Make files for `dataset_2`
    for fmt in ["jsonl", "json", "csv"]:
        dataset = os.path.join(
            storage, f"dataset_2__{request.keywords.node.name}.{fmt}"
        )
        if fmt == "jsonl":
            with open(dataset, "w", encoding="utf-8") as f:
                for item in dataset_2:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif fmt == "json":
            with open(dataset, "w", encoding="utf-8") as f:
                json.dump(dataset_2, f, ensure_ascii=False, indent=4)
        else:  # fmt == "csv"
            df = pd.DataFrame(dataset_2)
            df.to_csv(dataset, index=False)
        paths[f"dataset_2__{fmt}"] = dataset

    # Make file for `result_df_5_2`
    save_path = os.path.join(
        storage, f"result_df_5_2__{request.keywords.node.name}.csv"
    )
    result_df_5_2.reset_index(names="i").to_csv(save_path, index=False)
    paths["result_df_5_2"] = save_path

    # Make file for `openai_configuration`
    save_path = os.path.join(
        storage, f"openai_configuration__{request.keywords.node.name}.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(openai_configuration, f, indent=4, ensure_ascii=False)
    paths["openai_configuration"] = save_path

    # Make file for `openai_configuration_1w`
    save_path = os.path.join(
        storage, f"openai_configuration_1w__{request.keywords.node.name}.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(openai_configuration_1w, f, indent=4, ensure_ascii=False)
    paths["openai_configuration_1w"] = save_path

    return paths


#######################################################################################
#                                                                                     #
#                                  PREPARATION WORK                                   #
#                                                                                     #
#######################################################################################


class NormalBaseEvaluator(BaseEvaluator):
    """Extends the base evaluator.

    This can return random scores, fail on all items, or fail on a certain item based
    on the instruction field.
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

    def _get_score(self, data_item, **kwargs):
        if self.mode == "random" or self.mode.startswith("err_on_"):
            if (
                self.mode.startswith("err_on_instruction_")
                and data_item["instruction"] == self.mode[19:]
            ):
                raise ValueError
            if len(self.subjects) == 1:
                return random.randint(0, 100)
            return {subject: random.randint(0, 100) for subject in self.subjects}
        elif self.mode == "all_err":
            raise ValueError

    async def _aget_score(self, data_item, **kwargs):
        await asyncio.sleep(0.01)
        return self._get_score(data_item, **kwargs)


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
        self.lock = threading.Lock()
        self.tracker = {}

    def _get_score(self, data_item, **kwargs):
        instruction = data_item["instruction"]
        with self.lock:
            if instruction in self.tracker:
                ind = self.tracker[instruction]
                self.tracker[instruction] += 1
            else:
                ind = 0
                self.tracker[instruction] = 1
        return {
            subject: self.sc_mapping(ind, i) for i, subject in enumerate(self.subjects)
        }

    async def _aget_score(self, data_item, **kwargs):
        await asyncio.sleep(0.01)
        return self._get_score(data_item, **kwargs)


class NormalBaseOpenAIEvaluator(BaseOpenAIEvaluator):
    """Extends the base OpenAI evaluator.

    This can judges the correctness of response based on whether the response is ``in``
    the reference answer.
    """

    _pattern = re.compile(r"```\n(.+)\n```\n\n```\n(.+)\n```", re.DOTALL)

    def __init__(
        self,
        dataset,
        save_path,
        subjects,
        openai_config,
        *,
        fmt="jsonl",
        n_iter=1,
        agg_method=None,
        timeout=60,
        model="gpt-3.5-turbo",
        logging_mode="all",
        verbose=1,
        **openai_kwargs,
    ):
        super().__init__(
            dataset,
            save_path,
            subjects,
            openai_config,
            fmt=fmt,
            n_iter=n_iter,
            agg_method=agg_method,
            timeout=timeout,
            model=model,
            logging_mode=logging_mode,
            verbose=verbose,
            **openai_kwargs,
        )

    def _prompt(self, data_item):
        reference, actual = data_item["output"], data_item["response"]
        return "", f"```\n{reference}\n```\n\n```\n{actual}\n```"

    def _extract_scores(self, reply):
        return int(reply)


def mock_openai_chatcompletion_create(*args, **kwargs):
    """Mock function for openai.ChatCompletion.create."""
    for item in kwargs["messages"]:
        if item["role"] == "user":
            target_msg = item["content"]
            break

    mat = re.match(r"```\n(.+)\n```\n\n```\n(.+)\n```", target_msg, re.DOTALL)
    reference, actual = mat.group(1), mat.group(2)
    score = (actual in reference) * 100

    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": str(score), "role": "assistant"},
            }
        ],
        "usage": {f"{term}_tokens": 1 for term in ["completion", "prompt", "total"]},
    }


async def mock_openai_chatcompletion_acreate(*args, **kwargs):
    """Mock function for openai.ChatCompletion.acreate."""
    await asyncio.sleep(0.01)
    return mock_openai_chatcompletion_create(*args, **kwargs)


def mock_openai_chatcompletion_create_fail2stop(*args, **kwargs):
    """Mock function for openai.ChatCompletion.create."""
    return {
        "choices": [
            {
                "finish_reason": "null",
                "message": {"content": "Hello.", "role": "assistant"},
            }
        ],
        "usage": {f"{term}_tokens": 1 for term in ["completion", "prompt", "total"]},
    }


async def mock_openai_chatcompletion_acreate_fail2stop(*args, **kwargs):
    """Mock function for openai.ChatCompletion.acreate."""
    await asyncio.sleep(0.01)
    return mock_openai_chatcompletion_create_fail2stop(*args, **kwargs)


#######################################################################################
#                                                                                     #
#                                  TESTS START HERE                                   #
#                                                                                     #
#######################################################################################


class TestBaseEvaluator:
    """Testing ml3m.base.BaseEvaluator."""

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize("subjects", [["score1"], ["score2", "score3"]])
    @pytest.mark.parametrize("workers", [1, 3])
    @pytest.mark.parametrize("n_iter,agg_method", [(1, None), (3, "sum"), (3, "mode")])
    @pytest.mark.parametrize("logging_mode", ["none", "all", "failed"])
    @pytest.mark.parametrize("verbose", [0, 1, 2])
    def test_base_evaluator_result_versus_written(
        self,
        fmt,
        subjects,
        workers,
        n_iter,
        agg_method,
        logging_mode,
        verbose,
        storage,
        prepare,
        request,
    ):
        """Test that evaluator._result_df and the written csv are the same.

        This serves as a basis so that the other tests do not need to I/O to check the
        results but instead directly read the attribute.
        """
        dataset = prepare[f"dataset_2__{fmt}"]
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
            logging_mode=logging_mode,
            verbose=verbose,
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
        self, fmt, subjects, new_subjects, workers, storage, prepare, request
    ):
        """Test the basic evaluator functionalities.

        Fail all data items
        -> Pass one of the data items
        -> Pass all data items
        -> Evaluate again (should make no change)
        -> Fail all data items on the new subjects (should make no change)
        -> Pass one of the data items on the new subjects
        -> Evaluate on old subject(s) again (should make no change)
        -> Pass all data items on the new subjects
        -> Evaluate again (should make no change)
        """
        dataset = prepare[f"dataset_2__{fmt}"]
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
        assert list(df.index) == [0, 1]
        assert list(df.columns) == subjects
        assert df.isna().all().all()

        # This should pass the second data item but fail the first
        item_0_instruction = dataset_2[0]["instruction"]
        evaluator.mode = f"err_on_instruction_{item_0_instruction}"
        completed = evaluator.evaluate()
        assert not completed

        df = evaluator._result_df
        item_1_scores = df.loc[1, :]
        assert not item_1_scores.isna().any()

        # This should pass all of the data items
        evaluator.mode = "random"
        completed = evaluator.evaluate()
        assert completed

        df = evaluator._result_df
        items_scores = df.copy()
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
        assert list(df.columns) == [*subjects, *new_subjects]
        assert df.loc[:, new_subjects].isna().all().all()
        assert_frame_equal(items_scores, df.loc[:, subjects])

        # This should pass the second data item but fail the first on the new subjects
        evaluator2.mode = f"err_on_instruction_{item_0_instruction}"
        completed = evaluator2.evaluate()
        assert not completed

        df = evaluator2._result_df
        item_1_scores = df.loc[1, :]
        new_items_scores = df.copy()
        assert df.loc[0, new_subjects].isna().all()
        assert not df.loc[1, new_subjects].isna().any()
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
    @pytest.mark.parametrize("subjects", [["score1"], ["score2", "score3"]])
    @pytest.mark.parametrize("new_subjects", [["score4"], ["score5", "score6"]])
    @pytest.mark.parametrize("workers", [1, 3])
    def test_base_evaluator_overwrite(
        self, fmt, subjects, new_subjects, workers, storage, prepare, request
    ):
        """Test overwrite parameter of evaluate.

        Pass all data items
        -> Overwrite and fail all data items
        -> Overwrite and pass all data items
        -> Overwrite and pass all data items on the new subjects
        -> Overwrite and fail all data items on the new subjects
        -> Pass all data items on the new subjects
        -> Overwrite and pass all data items on the new subjects
        """
        dataset = prepare[f"dataset_2__{fmt}"]
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        # Preparation: Pass all data items
        evaluator = NormalBaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=subjects,
            fmt=fmt,
            workers=workers,
        )
        evaluator.evaluate()
        df_ref = evaluator._result_df

        # Overwrite and fail all data items
        evaluator.mode = "all_err"
        completed = evaluator.evaluate(overwrite=True)
        assert not completed

        df = evaluator._result_df
        assert df.isna().all().all()

        # Overwrite and pass all data items
        evaluator.mode = "random"
        evaluator.evaluate(overwrite=True)
        df_ref = evaluator._result_df

        # Overwrite and pass all data items on the new subjects
        completed = evaluator.evaluate(overwrite=True)
        assert completed

        df = evaluator._result_df
        with pytest.raises(AssertionError):
            assert_frame_equal(df, df_ref)
        df_ref = df

        # Overwrite and fail all data items on the new subjects
        evaluator2 = NormalBaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=new_subjects,
            fmt=fmt,
            workers=workers,
            mode="all_err",
        )
        completed = evaluator2.evaluate(overwrite=True)
        assert not completed

        df = evaluator2._result_df
        assert_frame_equal(df.loc[:, subjects], df_ref)
        assert df.loc[:, new_subjects].isna().all().all()

        # Preparation: Pass all data items on the new subjects
        evaluator2.mode = "random"
        evaluator2.evaluate()
        df_ref = evaluator2._result_df

        # Overwrite and pass all data items on the new subjects
        completed = evaluator2.evaluate(overwrite=True)
        assert completed

        df = evaluator2._result_df
        with pytest.raises(AssertionError):
            assert_frame_equal(df, df_ref)

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize("workers", [1, 3])
    @pytest.mark.parametrize("agg_method", ["mean", "sum", "min", "max", "mode"])
    def test_base_evaluator_aggregate(
        self, fmt, workers, agg_method, storage, prepare, request
    ):
        """Test the aggregation methods."""
        subjects = ["score1", "score2"]
        dataset = prepare[f"dataset_2__{fmt}"]
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
    def test_base_evalutor_overlapping_subjects(self, fmt, storage, prepare, request):
        """Test the behavior when two evaluations have overlapping subject.

        The overlapped subject will be overwritten without any warning or error.
        """
        dataset = prepare[f"dataset_2__{fmt}"]
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

        evaluator = NormalBaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=["score"],
            fmt=fmt,
        )
        completed = evaluator.evaluate()
        assert completed
        assert_frame_equal(evaluator._result_df, df)

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

        # score1 will stay the same, but score should be overwritten even if overwrite
        # is not specified; this is because score2 is missing while score and score2
        # are treated as a whole in the second evaluator
        # Since the evaluator is in random mode, most probably the overwritten part
        # will not be the same as what it previously was
        assert_series_equal(df.loc[:, "score1"], df2.loc[:, "score1"])
        with pytest.raises(AssertionError):
            assert_series_equal(df.loc[:, "score"], df2.loc[:, "score"])

    def test_base_evalutor_invalid_init(self, storage, prepare, request):
        """Test invalid initialization."""
        dataset = prepare["dataset_2__jsonl"]
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

    def test_base_evalutor_check_score(self, storage, prepare, request):
        """Test the `_check_score` method."""
        dataset = prepare["dataset_2__jsonl"]
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        evaluator_1 = BaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=["score"],
        )
        evaluator_3 = BaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=["score1", "score2", "score3"],
        )

        # Test score not real or dict
        for evaluator, scores in product(
            [evaluator_1, evaluator_3], [None, complex(1, 2), pd.NA]
        ):
            msg = re.escape(
                "The scores must be either a real number or a dictionary with real "
                f"values; got '{scores}' of type '{type(scores)}' instead."
            )
            with pytest.raises(ScoringError, match=msg):
                evaluator._check_scores(scores)

        # Test score dict with not all values being real
        for invalid_val in [None, complex(1, 2), pd.NA]:
            msg = re.escape(
                "The scores must be either a real number or a dictionary with real "
                f"values; got a dictionary but there exists 'score: {invalid_val}' of "
                f"type '{type(invalid_val)}'."
            )
            with pytest.raises(ScoringError, match=msg):
                evaluator_1._check_scores({"score": invalid_val})
            msg = re.escape(
                "The scores must be either a real number or a dictionary with real "
                f"values; got a dictionary but there exists 'score2: {invalid_val}' of "
                f"type '{type(invalid_val)}'."
            )
            with pytest.raises(ScoringError, match=msg):
                evaluator_3._check_scores(
                    {"score1": 1, "score2": invalid_val, "score3": invalid_val}
                )

        # Test exception when single score but multiple subjects
        msg = re.escape("Got a single score but multiple subjects are assigned.")
        with pytest.raises(ScoringError, match=msg):
            evaluator_3._check_scores(10)

        # Test exception when missing required subjects
        msg = re.compile(r"Missing keys: '.+', '.+'\.")
        with pytest.raises(ScoringError, match=msg):
            evaluator_3._check_scores({"score2": 10, "score4": 10})

        # Test normal functionalities
        for scores in [10, {"score": 10}, {"score": 10, "extra_score": 10}]:
            assert evaluator_1._check_scores(scores) == {"score": 10}
        expected = {"score1": 10, "score2": 10, "score3": 10}
        scores = expected.copy()
        assert evaluator_3._check_scores(scores) == expected
        scores["score4"] = 10
        assert evaluator_3._check_scores(scores) == expected

    @pytest.mark.parametrize("subject_subset", [None, ["score2"], ["score2", "score1"]])
    @pytest.mark.parametrize("items", [None, [3, 5, 0], list(range(7))])
    def test_base_evalutor_scoring_methods(self, subject_subset, items, prepare):
        """Test the `load_scores` and `load_avg_score` methods."""
        dataset = prepare["dataset_2__jsonl"]
        save_path = prepare["result_df_5_2"]

        evaluator = BaseEvaluator(
            dataset=dataset,
            save_path=save_path,
            subjects=["score1", "score2"],
        )

        expected_df = result_df_5_2
        if subject_subset is not None:
            expected_df = expected_df[subject_subset]
        if items is not None:
            expected_df = expected_df.reindex(items)

        subjects = ["score1", "score2"] if subject_subset is None else subject_subset
        expected_scores = {
            subject: np.mean(expected_df.loc[:, subject].dropna())
            for subject in subjects
        }

        result_df = evaluator.load_scores(subject_subset=subject_subset, items=items)
        result_scores = evaluator.load_avg_score(
            subject_subset=subject_subset, items=items
        )

        assert_frame_equal(result_df, expected_df, check_dtype=False)
        assert expected_scores == result_scores


class TestBaseOpenAIEvaluator:
    """Testing ml3m.base.BaseOpenAIEvaluator."""

    @pytest.mark.parametrize(
        "config_name", ["openai_configuration", "openai_configuration_1w"]
    )
    def test_base_openai_evaluator_evaluate_basics(
        self, config_name, storage, prepare, mocker, request
    ):
        """Test the basic evaluator functionalities."""
        dataset = prepare["dataset_2__jsonl"]
        openai_config = prepare[config_name]
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        patcher_create = mocker.patch("ml3m.base.eval.openai.ChatCompletion.create")
        patcher_acreate = mocker.patch("ml3m.base.eval.openai.ChatCompletion.acreate")
        evaluator = NormalBaseOpenAIEvaluator(
            dataset=dataset,
            save_path=save_path,
            openai_config=openai_config,
            subjects=["score"],
        )

        # Patch to always fail with stop_reason != "stop"
        patcher_create.side_effect = mock_openai_chatcompletion_create_fail2stop
        patcher_acreate.side_effect = mock_openai_chatcompletion_acreate_fail2stop
        completed = evaluator.evaluate()
        assert not completed

        df = evaluator._result_df
        expected = pd.DataFrame(index=pd.Index([0, 1], name="i"), columns=["score"])
        expected["score"] = None
        assert_frame_equal(df, expected, check_dtype=False)

        # Patch to respond normally
        patcher_create.side_effect = mock_openai_chatcompletion_create
        patcher_acreate.side_effect = mock_openai_chatcompletion_acreate
        completed = evaluator.evaluate()
        assert completed

        df = evaluator._result_df
        expected.at[0, "score"] = 100
        expected.at[1, "score"] = 0
        assert_frame_equal(df, expected, check_dtype=False)

    def test_base_openai_evaluator_invalid_init(self, storage, prepare, request):
        """Test invalid initialization."""
        dataset = prepare["dataset_2__jsonl"]
        openai_config = prepare["openai_configuration"]
        save_path = os.path.join(
            storage, f"save_path__{request.keywords.node.name}.csv"
        )

        for openai_kwargs in [
            {"api_key": "xxx"},
            {"api_base": "xxx"},
            {"api_key": "xxx", "api_base": "xxx"},
        ]:
            with pytest.raises(InvalidParameterError):
                NormalBaseOpenAIEvaluator(
                    dataset=dataset,
                    save_path=save_path,
                    openai_config=openai_config,
                    subjects=["score"],
                    **openai_kwargs,
                )
