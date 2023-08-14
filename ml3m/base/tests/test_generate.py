import asyncio
import json
import os
from functools import partial

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml3m.base import ResponseGenerator

#######################################################################################
#                                                                                     #
#                                        DATA                                         #
#                                                                                     #
#######################################################################################


orig_dataset_2 = [
    {
        "instruction": "What is the capital of China?",
        "input": "",
        "output": "The capital of China is Beijing.",
    },
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris.",
    },
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

    return paths


#######################################################################################
#                                                                                     #
#                                  PREPARATION WORK                                   #
#                                                                                     #
#######################################################################################


def query_func_fixed(data_item, response):
    """Return a fixed response."""
    return response


async def query_afunc_fixed(data_item, response):
    """Return a fixed response."""
    await asyncio.sleep(0.01)
    return query_func_fixed(data_item, response)


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
    @pytest.mark.parametrize("verbose", [0, 1, 2])
    def test_base_generator_result_versus_written(
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
        """Test that generator._all_data and the written dataset are the same."""
        orig_dataset = prepare[f"orig_dataset_2__{fmt}"]
        dataset = os.path.join(storage, f"dataset__{request.keywords.node.name}.{fmt}")

        generator = ResponseGenerator(
            orig_dataset=orig_dataset,
            dataset=dataset,
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
