import asyncio
import copy
import json
import os
import re
from functools import partial

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml3m.base import ResponseGenerator
from ml3m.errors import InvalidParameterError

#######################################################################################
#                                                                                     #
#                                        DATA                                         #
#                                                                                     #
#######################################################################################


orig_dataset_2 = [
    {
        "instruction": "What is the capital of China?",
        "output": "The capital of China is Beijing.",
    },
    {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris.",
    },
]

orig_dataset_l2 = [
    ["What is the capital of China?", "The capital of China is Beijing."],
    ["What is the capital of France?", "The capital of France is Paris."],
]


@pytest.fixture(scope="module")
def prepare(request, storage):
    """Make a temporary storage and clear it towards the end."""
    paths = {}

    # Make files for `orig_dataset_2`
    for fmt in ["jsonl", "json", "csv"]:
        dataset = os.path.join(
            storage, f"orig_dataset_2__{request.keywords.node.name}.{fmt}"
        )
        if fmt == "jsonl":
            with open(dataset, "w", encoding="utf-8") as f:
                for item in orig_dataset_2:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif fmt == "json":
            with open(dataset, "w", encoding="utf-8") as f:
                json.dump(orig_dataset_2, f, ensure_ascii=False, indent=4)
        else:  # fmt == "csv"
            df = pd.DataFrame(orig_dataset_2)
            df.to_csv(dataset, index=False)
        paths[f"orig_dataset_2__{fmt}"] = dataset

    # Make files for `orig_dataset_l2`
    for fmt in ["jsonl", "json"]:
        dataset = os.path.join(
            storage, f"orig_dataset_l2__{request.keywords.node.name}.{fmt}"
        )
        with open(dataset, "w", encoding="utf-8") as f:
            if fmt == "jsonl":
                for item in orig_dataset_l2:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:  # fmt == "json":
                json.dump(orig_dataset_l2, f, ensure_ascii=False, indent=4)
        paths[f"orig_dataset_l2__{fmt}"] = dataset

    return paths


#######################################################################################
#                                                                                     #
#                                  PREPARATION WORK                                   #
#                                                                                     #
#######################################################################################


def query_func_fixed(query, response="", mode="normal"):
    """Return a fixed response.

    This can normally pass all items, fail all items, fail a certain item based on the
    instruction field, or fail a certain item based on the a certain index (when each
    item is a list).
    """
    if mode == "normal" or mode.startswith("err_on_"):
        if mode.startswith("err_on_instruction_") and query["instruction"] == mode[19:]:
            raise ValueError
        mat = re.match(r"err_on_index\.(\d+)_(.+)", mode, re.DOTALL)
        if mat is not None:
            if isinstance(query, list) and query[int(mat.group(1))] == mat.group(2):
                raise ValueError
            if (
                isinstance(query, dict)
                and "data" in query
                and isinstance(query["data"], list)
                and query["data"][int(mat.group(1))] == mat.group(2)
            ):
                raise ValueError
        return response
    elif mode == "all_err":
        raise ValueError


async def query_afunc_fixed(query, response, mode="normal"):
    """Return a fixed response.

    This can normally pass all items, fail all items, fail a certain item based on the
    instruction field, or fail a certain item based on the a certain index (when each
    item is a list).
    """
    await asyncio.sleep(0.01)
    return query_func_fixed(query, response, mode=mode)


#######################################################################################
#                                                                                     #
#                                  TESTS START HERE                                   #
#                                                                                     #
#######################################################################################


class TestResponseGenerator:
    """Testing ml3m.base.ResponseGenerator."""

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize(
        "n_workers,query_func",
        [
            (1, partial(query_func_fixed, response="I don't know.")),
            (3, partial(query_afunc_fixed, response="I don't know.")),
        ],
    )
    @pytest.mark.parametrize("response_name", ["response", "model_response"])
    @pytest.mark.parametrize("logging_mode", ["none", "all", "failed"])
    @pytest.mark.parametrize("verbose", [-1, 0, 1, 2])
    def test_response_generator_result_versus_written(
        self,
        query_func,
        response_name,
        fmt,
        n_workers,
        logging_mode,
        verbose,
        storage,
        prepare,
        request,
    ):
        """Test that generator._all_data and the written dataset are the same.

        This serves as a basis so that the other tests do not need to I/O to check the
        results but instead directly read the attribute.
        """
        orig_dataset = prepare[f"orig_dataset_2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        generator = ResponseGenerator(
            orig_dataset=orig_dataset,
            dataset=dataset,
            info_func=lambda x: x,
            query_func=query_func,
            response_name=response_name,
            fmt=fmt,
            n_workers=n_workers,
            logging_mode=logging_mode,
            verbose=verbose,
        )
        completed = generator.generate()
        assert completed

        if fmt == "jsonl":
            with open(dataset, "r", encoding="utf-8") as f:
                data_saved = [json.loads(line) for line in f]
            assert data_saved == generator._all_data

        elif fmt == "json":
            with open(dataset, "r", encoding="utf-8") as f:
                data_saved = json.load(f)
            assert data_saved == generator._all_data

        else:  # fmt == "csv"
            data_saved = pd.read_csv(dataset)
            assert_frame_equal(data_saved, generator._all_data)

    @pytest.mark.parametrize("fmt", ["jsonl", "json"])
    @pytest.mark.parametrize(
        "n_workers,query_func",
        [
            (1, partial(query_func_fixed, response="I don't know.")),
            (3, partial(query_afunc_fixed, response="I don't know.")),
        ],
    )
    @pytest.mark.parametrize("response_name", ["response", "model_response"])
    @pytest.mark.parametrize("logging_mode", ["none", "all", "failed"])
    @pytest.mark.parametrize("verbose", [-1, 0, 1, 2])
    def test_response_generator_result_versus_written_list(
        self,
        query_func,
        response_name,
        fmt,
        n_workers,
        logging_mode,
        verbose,
        storage,
        prepare,
        request,
    ):
        """Test that generator._all_data and the written dataset are the same.

        This serves as a basis so that the other tests do not need to I/O to check the
        results but instead directly read the attribute.

        This is the special case where each data item is a list.
        """
        orig_dataset = prepare[f"orig_dataset_l2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        generator = ResponseGenerator(
            orig_dataset=orig_dataset,
            dataset=dataset,
            info_func=lambda x: x,
            query_func=query_func,
            response_name=response_name,
            fmt=fmt,
            n_workers=n_workers,
            logging_mode=logging_mode,
            verbose=verbose,
        )
        completed = generator.generate()
        assert completed

        with open(dataset, "r", encoding="utf-8") as f:
            if fmt == "jsonl":
                data_saved = [json.loads(line) for line in f]
            else:  # fmt == "json"
                data_saved = json.load(f)
        assert data_saved == generator._all_data

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize(
        "n_workers,query_func",
        [
            (1, partial(query_func_fixed, response="I don't know.")),
            (3, partial(query_afunc_fixed, response="I don't know.")),
        ],
    )
    def test_response_generator_generate_basics(
        self, fmt, n_workers, query_func, storage, prepare, request
    ):
        """Test the basic generator functionalities.

        Fail all data items
        -> Pass one of the data items
        -> Pass all data items
        -> Generate again (should make no change)
        -> Fail all data items with the new response name (should make no change)
        -> Pass one of the data items with the new response name
        -> Generate on the old response name again (should make no change)
        -> Pass all data items with the new response name
        -> Generate again (should make no change)
        """
        orig_dataset = prepare[f"orig_dataset_2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        def make_generator_by_mode(mode, response_name):
            """Convenient function for making a generator based on mode."""
            return ResponseGenerator(
                orig_dataset=orig_dataset,
                dataset=dataset,
                info_func=lambda x: x,
                query_func=partial(query_func, mode=mode),
                response_name=response_name,
                fmt=fmt,
                n_workers=n_workers,
            )

        # This should pass none of the data items
        generator = make_generator_by_mode("all_err", "response")
        completed = generator.generate()
        assert not completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            expected = copy.deepcopy(orig_dataset_2)
            expected[0]["response"] = None
            expected[1]["response"] = None
            assert results == expected
        else:  # fmt == "csv"
            expected_df = pd.DataFrame(orig_dataset_2)
            expected_df["response"] = None
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should pass the second data item but fail the firrst
        item_0_instruction = orig_dataset_2[0]["instruction"]
        generator = make_generator_by_mode(
            f"err_on_instruction_{item_0_instruction}", "response"
        )
        completed = generator.generate()
        assert not completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[1]["response"] = "I don't know."
            assert results == expected
        else:  # fmt == "csv"
            expected_df.at[1, "response"] = "I don't know."
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should pass all of the data items
        generator = make_generator_by_mode("normal", "response")
        completed = generator.generate()
        assert completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["response"] = "I don't know."
            assert results == expected
        else:  # fmt == "csv"
            expected_df.at[0, "response"] = "I don't know."
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should not modify the generated dataset
        generator.generate()
        completed = generator.generate()
        assert completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            assert results == expected
        else:  # fmt == "csv"
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should pass none of the data items with the new response name
        generator2 = make_generator_by_mode("all_err", "new_response")
        completed = generator2.generate()
        assert not completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["new_response"] = None
            expected[1]["new_response"] = None
            assert results == expected
        else:  # fmt == "csv"
            expected_df["new_response"] = None
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should pass the second item but fail the first with the new response
        # name
        generator2 = make_generator_by_mode(
            f"err_on_instruction_{item_0_instruction}", "new_response"
        )
        completed = generator2.generate()
        assert not completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[1]["new_response"] = "I don't know."
            assert results == expected
        else:  # fmt == "csv"
            expected_df.at[1, "new_response"] = "I don't know."
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should not modify the generated dataset since it is using the old
        # response name
        completed = generator.generate()
        assert completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            assert results == expected
        else:  # fmt == "csv"
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should pass all of the data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response")
        completed = generator2.generate()
        assert completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["new_response"] = "I don't know."
            assert results == expected
        else:  # fmt == "csv"
            expected_df.at[0, "new_response"] = "I don't know."
            assert_frame_equal(expected_df, results, check_dtype=False)

        # This should not modify the generated dataset
        completed = generator2.generate()
        assert completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            assert results == expected
        else:  # fmt == "csv"
            assert_frame_equal(expected_df, results, check_dtype=False)

    @pytest.mark.parametrize("fmt", ["jsonl", "json"])
    @pytest.mark.parametrize(
        "n_workers,query_func",
        [
            (1, partial(query_func_fixed, response="I don't know.")),
            (3, partial(query_afunc_fixed, response="I don't know.")),
        ],
    )
    def test_response_generator_generate_basics_list(
        self, fmt, n_workers, query_func, storage, prepare, request
    ):
        """Test the basic generator functionalities.

        Fail all data items
        -> Pass one of the data items
        -> Pass all data items
        -> Generate again (should make no change)
        -> Fail all data items with the new response name (should make no change)
        -> Pass one of the data items with the new response name
        -> Generate on the old response name again (should make no change)
        -> Pass all data items with the new response name
        -> Generate again (should make no change)

        This is the special case where each data item is a list.
        """
        orig_dataset = prepare[f"orig_dataset_l2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        def make_generator_by_mode(mode, response_name):
            """Convenient function for making a generator based on mode."""
            return ResponseGenerator(
                orig_dataset=orig_dataset,
                dataset=dataset,
                info_func=lambda x: x,
                query_func=partial(query_func, mode=mode),
                response_name=response_name,
                fmt=fmt,
                n_workers=n_workers,
            )

        # This should pass none of the data items
        generator = make_generator_by_mode("all_err", "response")
        completed = generator.generate()
        assert not completed

        results = generator._all_data
        expected = [
            {"data": orig_dataset_l2[0], "response": None},
            {"data": orig_dataset_l2[1], "response": None},
        ]
        assert results == expected

        # This should pass the second data item but fail the firrst
        item_0_instruction = orig_dataset_l2[0][0]
        generator = make_generator_by_mode(
            f"err_on_index.0_{item_0_instruction}", "response"
        )
        completed = generator.generate()
        assert not completed

        results = generator._all_data
        expected[1]["response"] = "I don't know."
        assert results == expected

        # This should pass all of the data items
        generator = make_generator_by_mode("normal", "response")
        completed = generator.generate()
        assert completed

        results = generator._all_data
        expected[0]["response"] = "I don't know."
        assert results == expected

        # This should not modify the generated dataset
        generator.generate()
        completed = generator.generate()
        assert completed

        results = generator._all_data
        assert results == expected

        # This should pass none of the data items with the new response name
        generator2 = make_generator_by_mode("all_err", "new_response")
        completed = generator2.generate()
        assert not completed

        results = generator2._all_data
        expected[0]["new_response"] = None
        expected[1]["new_response"] = None
        assert results == expected

        # This should pass the second item but fail the first with the new response
        # name
        generator2 = make_generator_by_mode(
            f"err_on_index.0_{item_0_instruction}", "new_response"
        )
        completed = generator2.generate()
        assert not completed

        results = generator2._all_data
        expected[1]["new_response"] = "I don't know."
        assert results == expected

        # This should not modify the generated dataset since it is using the old
        # response name
        completed = generator.generate()
        assert completed

        results = generator._all_data
        assert results == expected

        # This should pass all of the data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response")
        completed = generator2.generate()
        assert completed

        results = generator2._all_data
        expected[0]["new_response"] = "I don't know."
        assert results == expected

        # This should not modify the generated dataset
        completed = generator2.generate()
        assert completed

        results = generator2._all_data
        assert results == expected

    @pytest.mark.parametrize("fmt", ["jsonl", "json", "csv"])
    @pytest.mark.parametrize(
        "n_workers,query_func",
        [(1, query_func_fixed), (3, query_afunc_fixed)],
    )
    def test_response_generator_overwrite(
        self, fmt, n_workers, query_func, storage, prepare, request
    ):
        """Test overwrite parameter of evaluate.

        Pass all data items
        -> Overwrite and fail all data items
        -> Overwrite and pass all data items
        -> Overwrite and pass all data items with the new response name
        -> Overwrite and fail all data items with the new response name
        -> Pass all data items with the new response name
        -> Overwrite and pass all data items with the new response name
        """
        orig_dataset = prepare[f"orig_dataset_2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        def make_generator_by_mode(mode, response_name, response):
            """Convenient function for making a generator based on mode."""
            return ResponseGenerator(
                orig_dataset=orig_dataset,
                dataset=dataset,
                info_func=lambda x: x,
                query_func=partial(query_func, response=response, mode=mode),
                response_name=response_name,
                fmt=fmt,
                n_workers=n_workers,
            )

        # Preparation: Pass all data items
        generator = make_generator_by_mode("normal", "response", "a")
        generator.generate()

        # Overwrite and fail all data items
        generator = make_generator_by_mode("all_err", "response", "b")
        completed = generator.generate(overwrite=True)
        assert not completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            expected = copy.deepcopy(orig_dataset_2)
            expected[0]["response"] = None
            expected[1]["response"] = None
            assert results == expected
        else:  # fmt == "csv"
            expected_df = pd.DataFrame(orig_dataset_2)
            expected_df["response"] = None
            assert_frame_equal(expected_df, results, check_dtype=False)

        # Overwrite and pass all data items
        generator = make_generator_by_mode("normal", "response", "b")
        completed = generator.generate(overwrite=True)
        assert completed

        results = generator._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["response"] = "b"
            expected[1]["response"] = "b"
            assert results == expected
        else:  # fmt == "csv"
            expected_df["response"] = "b"
            assert_frame_equal(expected_df, results, check_dtype=False)

        # Overwrite and pass all data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response", "c")
        completed = generator2.generate(overwrite=True)
        assert completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["new_response"] = "c"
            expected[1]["new_response"] = "c"
            assert results == expected
        else:  # fmt == "csv"
            expected_df["new_response"] = "c"
            assert_frame_equal(expected_df, results, check_dtype=False)

        # Overwrite and fail all data items with the new response name
        generator2 = make_generator_by_mode("all_err", "new_response", "d")
        completed = generator2.generate(overwrite=True)
        assert not completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["new_response"] = None
            expected[1]["new_response"] = None
            assert results == expected
        else:  # fmt == "csv"
            expected_df["new_response"] = None
            assert_frame_equal(expected_df, results, check_dtype=False)

        # Preparation: Pass all data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response", "d")
        generator2.generate()

        # Overwrite and pass all data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response", "e")
        completed = generator2.generate(overwrite=True)
        assert completed

        results = generator2._all_data
        if fmt == "json" or fmt == "jsonl":
            expected[0]["new_response"] = "e"
            expected[1]["new_response"] = "e"
            assert results == expected
        else:  # fmt == "csv"
            expected_df["new_response"] = "e"
            assert_frame_equal(expected_df, results, check_dtype=False)

    @pytest.mark.parametrize("fmt", ["jsonl", "json"])
    @pytest.mark.parametrize(
        "n_workers,query_func",
        [(1, query_func_fixed), (3, query_afunc_fixed)],
    )
    def test_response_generator_overwrite_list(
        self, fmt, n_workers, query_func, storage, prepare, request
    ):
        """Test overwrite parameter of evaluate.

        Pass all data items
        -> Overwrite and fail all data items
        -> Overwrite and pass all data items
        -> Overwrite and pass all data items with the new response name
        -> Overwrite and fail all data items with the new response name
        -> Pass all data items with the new response name
        -> Overwrite and pass all data items with the new response name

        This is the special case where each data item is a list.
        """
        orig_dataset = prepare[f"orig_dataset_l2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        def make_generator_by_mode(mode, response_name, response):
            """Convenient function for making a generator based on mode."""
            return ResponseGenerator(
                orig_dataset=orig_dataset,
                dataset=dataset,
                info_func=lambda x: x,
                query_func=partial(query_func, response=response, mode=mode),
                response_name=response_name,
                fmt=fmt,
                n_workers=n_workers,
            )

        # Preparation: Pass all data items
        generator = make_generator_by_mode("normal", "response", "a")
        generator.generate()

        # Overwrite and fail all data items
        generator = make_generator_by_mode("all_err", "response", "b")
        completed = generator.generate(overwrite=True)
        assert not completed

        results = generator._all_data
        expected = [
            {"data": orig_dataset_l2[0], "response": None},
            {"data": orig_dataset_l2[1], "response": None},
        ]
        assert results == expected

        # Overwrite and pass all data items
        generator = make_generator_by_mode("normal", "response", "b")
        completed = generator.generate(overwrite=True)
        assert completed

        results = generator._all_data
        expected[0]["response"] = "b"
        expected[1]["response"] = "b"
        assert results == expected

        # Overwrite and pass all data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response", "c")
        completed = generator2.generate(overwrite=True)
        assert completed

        results = generator2._all_data
        expected[0]["new_response"] = "c"
        expected[1]["new_response"] = "c"
        assert results == expected

        # Overwrite and fail all data items with the new response name
        generator2 = make_generator_by_mode("all_err", "new_response", "d")
        completed = generator2.generate(overwrite=True)
        assert not completed

        results = generator2._all_data
        expected[0]["new_response"] = None
        expected[1]["new_response"] = None
        assert results == expected

        # Preparation: Pass all data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response", "d")
        generator2.generate()

        # Overwrite and pass all data items with the new response name
        generator2 = make_generator_by_mode("normal", "new_response", "e")
        completed = generator2.generate(overwrite=True)
        assert completed

        results = generator2._all_data
        expected[0]["new_response"] = "e"
        expected[1]["new_response"] = "e"
        assert results == expected

    def test_response_generator_invalid_init(self, storage, prepare, request):
        """Test invalid initialization."""
        orig_dataset = prepare["orig_dataset_2__jsonl"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.jsonl")

        # Test invalid n_workers
        for n_workers in [0, 2.4]:
            with pytest.raises(InvalidParameterError):
                ResponseGenerator(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=lambda x: x,
                    query_func=query_func_fixed,
                    response_name="response",
                    n_workers=n_workers,
                )

        # Test invalid info_func
        for info_func in [None, "func"]:
            with pytest.raises(InvalidParameterError):
                ResponseGenerator(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=info_func,
                    query_func=query_func_fixed,
                    response_name="response",
                )

        # Test invalid query_func
        for query_func in [None, "func"]:
            with pytest.raises(InvalidParameterError):
                ResponseGenerator(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=lambda x: x,
                    query_func=query_func,
                    response_name="response",
                )

        # Test incompatible n_workers with query_func
        for n_workers, query_func in zip(
            [1, 3],
            [query_afunc_fixed, query_func_fixed],
        ):
            with pytest.raises(InvalidParameterError):
                ResponseGenerator(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=lambda x: x,
                    query_func=query_func,
                    response_name="response",
                    n_workers=n_workers,
                )

        # Test invalid fmt
        for fmt in [".csv", "txt"]:
            with pytest.raises(InvalidParameterError):
                ResponseGenerator(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=lambda x: x,
                    query_func=query_func_fixed,
                    response_name="response",
                    fmt=fmt,
                )

        # Test invalid logging_mode
        for logging_mode in ["any", "succeeded"]:
            with pytest.raises(InvalidParameterError):
                ResponseGenerator(
                    orig_dataset=orig_dataset,
                    dataset=dataset,
                    info_func=lambda x: x,
                    query_func=query_func_fixed,
                    response_name="response",
                    logging_mode=logging_mode,
                )
