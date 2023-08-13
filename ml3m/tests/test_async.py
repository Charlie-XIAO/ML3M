import asyncio
import time
from functools import partial

import pytest

from ml3m._async import AsyncRunner

#######################################################################################
#                                                                                     #
#                                  PREPARATION WORK                                   #
#                                                                                     #
#######################################################################################


def process_func(item, **kwargs):
    """Pass all items."""
    return item * 10, f"Succeeded on {item}", None


async def process_afunc(item, addtlks=None, **kwargs):
    """Pass all items."""
    return process_func(item, **kwargs)


def failing_process_func(item, mode, **kwargs):
    """Fail items >= 15, supporting different modes of failing."""
    if item >= 15:
        if mode == "proper":
            return None, None, f"Failed on {item}"
        elif mode == "exception":
            raise ValueError
        elif mode == "single_val":
            return 10
        elif mode == "not3_val":
            return 10, None
    return item * 10, f"Succeeded on {item}", None


async def failing_process_afunc(item, mode, addtlks=None, **kwargs):
    """Fail items >= 15, supporting different modes of failing."""
    return failing_process_func(item, mode, **kwargs)


#######################################################################################
#                                                                                     #
#                                  TESTS START HERE                                   #
#                                                                                     #
#######################################################################################


class TestAsyncRunner:
    """Testing ml3m._async.AsyncRunner."""

    @pytest.mark.parametrize(
        "worker_kwargs",
        [
            [{}],
            [{"dummy": None}],
            [{} for _ in range(5)],
            [{"dummy": None} for _ in range(5)],
        ],
    )
    @pytest.mark.parametrize(
        "func,afunc,passed,failed",
        [(process_func, process_afunc, range(0, 200, 10), [])]
        + [
            (
                partial(failing_process_func, mode=mode),
                partial(failing_process_afunc, mode=mode),
                range(0, 150, 10),
                range(15, 20),
            )
            for mode in ["proper", "exception", "single_val", "not3_val"]
        ],
    )
    def test_async_runner_basics(self, func, afunc, passed, failed, worker_kwargs):
        """Test the basic functionalities of the asynchronous runner."""
        items = list(range(20))
        runner = AsyncRunner(process_func=func, process_afunc=afunc)

        results, failed_items = runner.run(items=items, worker_kwargs=worker_kwargs)
        assert set(results) == set(passed)
        assert set(failed_items) == set(failed)

    def test_async_runner_speedup(self):
        """Test that asynchronous parallelization speeds up."""
        items = [0.01] * 100

        def process_func(item, **kwargs):
            time.sleep(item)
            return item, f"Slept {item}s", None

        async def process_afunc(item, addtlks=None, **kwargs):
            await asyncio.sleep(item)
            return item, f"Slept {item}s", None

        runner = AsyncRunner(process_func=process_func, process_afunc=process_afunc)

        # Running with only a single worker
        s = time.time()
        runner.run(items=items, worker_kwargs=[{}])
        single_worker_time = time.time() - s

        # Running with ten workers
        s = time.time()
        runner.run(items=items, worker_kwargs=[{} for _ in range(10)])
        multi_worker_time = time.time() - s

        # Loosen the speedup: 10 workers, at least 5x speedup
        assert multi_worker_time < single_worker_time / 5
