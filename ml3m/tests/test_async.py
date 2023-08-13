import asyncio
import time
from functools import partial

import pytest

from ml3m._async import AsyncRunner
from ml3m.errors import InvalidParameterError

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
    await asyncio.sleep(0.01)
    return process_func(item, **kwargs)


def failing_process_func(item, mode, **kwargs):
    """Fail items >= 5, supporting different modes of failing."""
    if item >= 5:
        if mode == "proper":
            return None, None, f"Failed on {item}"
        elif mode == "exception":
            raise ValueError
        elif mode == "single_val":
            return item * 10
        elif mode == "not3_val":
            return item * 10, None
    return item * 10, f"Succeeded on {item}", None


async def failing_process_afunc(item, mode, addtlks=None, **kwargs):
    """Fail items >= 5, supporting different modes of failing."""
    await asyncio.sleep(0.01)
    return failing_process_func(item, mode, **kwargs)


def process_func_novarkey(item):
    """Accepts no variable-length keyword arguments."""


async def process_afunc_novarkey(item, addtlks=None):
    """Accepts no variable-length keyword arguments."""
    await asyncio.sleep(0.01)


async def process_afunc_noaddtlks(item, **kwargs):
    """Does not accept addtlks."""
    await asyncio.sleep(0.01)


async def process_afunc_addtlks_positional(item, addtlks, **kwargs):
    """Positional addtlks thus no default value."""
    await asyncio.sleep(0.01)


async def process_afunc_addtlks_default_not_none(item, addtlks=1, **kwargs):
    """Default value of addtlks not None."""
    await asyncio.sleep(0.01)


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
            [{}, {}],
            [{"dummy": None}, {"dummy": None}],
        ],
    )
    @pytest.mark.parametrize(
        "func,afunc,passed,failed",
        [(process_func, process_afunc, range(0, 100, 10), [])]
        + [
            (
                partial(failing_process_func, mode=mode),
                partial(failing_process_afunc, mode=mode),
                range(0, 50, 10),
                range(5, 10),
            )
            for mode in ["proper", "exception", "single_val", "not3_val"]
        ],
    )
    def test_async_runner_basics(self, func, afunc, passed, failed, worker_kwargs):
        """Test the basic functionalities of the asynchronous runner."""
        runner = AsyncRunner(process_func=func, process_afunc=afunc)

        results, failed_items = runner.run(
            items=list(range(10)), worker_kwargs=worker_kwargs
        )
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

    @pytest.mark.parametrize(
        "worker_kwargs,process_func,process_afunc",
        [
            ([{}], None, process_afunc),  # process_func is None, 1 worker
            ([{}], process_afunc, None),  # process_func is asynchronous
            ([{}, {}], process_func, None),  # process_afunc is None, >1 worker
            ([{}, {}], None, process_func),  # process_afunc is synchronous
            ([{}], process_func_novarkey, None),
            ([{}, {}], None, process_afunc_novarkey),
            ([{}, {}], None, process_afunc_noaddtlks),
            ([{}, {}], None, process_afunc_addtlks_positional),
            ([{}, {}], None, process_afunc_addtlks_default_not_none),
        ],
    )
    def test_async_runner_invalid_run(self, worker_kwargs, process_func, process_afunc):
        """Test invalid initialization and running arguments."""
        runner = AsyncRunner(
            process_func=process_func,
            process_afunc=process_afunc,
        )

        with pytest.raises(InvalidParameterError):
            runner.run(list(range(10)), worker_kwargs=worker_kwargs)
