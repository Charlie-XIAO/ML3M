"""This file is for asynchronous parallelization functionalities."""


from __future__ import annotations

import asyncio
from inspect import _ParameterKind, signature
from typing import Any, Callable, Iterable, NoReturn

from tqdm import tqdm

from ._color import COLOR, colored
from ._emoji import EMOJI


class AsyncRunner:
    """An asynchronous runner.

    Parameters
    ----------
    items : Iterable
        The items to process.
    worker_kwargs : list of dict
        The additional keyword arguments to pass into each asynchronous worker. The
        length of this list determines the number of workers to create.
    process_func : Callable
        The processing function that takes a data item then returns three things: the
        result, a normal message, and an error message. If the processing succeeds, the
        error message should be ``None``. If the processing errors out, the result and
        the normal message should be ``None``. Note that this function must be
        asynchrounous. Also, this function needs to accept ``**kwargs`` and a keyword
        argument ``addtlks`` of type ``list[asyncio.Lock] | None``.
    n_locks : int
        The additional locks to request. The requested locks will be passed to
        ``process_func`` via keyword argument ``addtlks``.
    verbose : int, default=1
        The verbosity level of the processing. For level 0, only a progress bar will be
        displayed. For level 1, the errored items will also be displayed. For levels
        higher than 2, all items will be displayed.
    """

    def __init__(
        self,
        items: Iterable,
        worker_kwargs: list[dict[str, Any]],
        process_func: Callable,
        n_locks: int = 0,
        verbose: int = 1,
    ):
        self.items = items
        self.worker_kwargs = worker_kwargs
        self.process_func = process_func
        self.n_locks = n_locks
        self.verbose = verbose
        self._async_mode = len(worker_kwargs) > 1

        # Validate the processing function (not exhaustive)
        sig = signature(self.process_func)
        if "addtlks" not in sig.parameters:
            raise ValueError("process_func must accept a keyword argument 'addtlks'.")
        elif sig.parameters["addtlks"].default is not None:
            raise ValueError(
                "Keyword argument 'addtlks' of process_func must be 'None' by default."
            )
        if not any(
            param.kind == _ParameterKind.VAR_KEYWORD
            for param in sig.parameters.values()
        ):
            raise ValueError("process_func must accept **kwargs.")

    async def _mainloop(self) -> None:
        """Main event loop for asynchronous parallelization."""
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        n_items = 0
        for item in self.items:
            self.queue.put_nowait(item)
            n_items += 1
        if n_items == 0:
            return

        # Create necessary asynchronous locks and additional ones on demand
        self.mainlk = asyncio.Lock()
        self.proglk = asyncio.Lock()
        self.addtlks = [asyncio.Lock() for _ in range(self.n_locks)]

        # Create worker tasks to process the queue asynchronously
        print(f"Initializing {len(self.worker_kwargs)} workers for {n_items} items...")
        tasks: list[asyncio.Task] = []
        self.progbar = tqdm(total=n_items)
        for worker_id, kwargs in enumerate(self.worker_kwargs):
            tasks.append(asyncio.create_task(self._worker(worker_id, **kwargs)))

        # Wait until the queue is fully processed and collect the results
        await self.queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.progbar.close()

    async def _worker(self, worker_id: int, **kwargs) -> NoReturn:
        """The worker for processing the asynchronous queue.

        Parameters
        ----------
        worker_id : int
            The id of the worker.
        """
        while True:
            item = await self.queue.get()
            result, norm_msg, err_msg = await self.process_func(
                item, addtlks=self.addtlks, **kwargs
            )

            # Print the execution information by demand
            prefix = f"[W{worker_id:03d}]"
            async with self.proglk:
                if result is not None and self.verbose >= 2:
                    tqdm.write(f"{prefix:<10} {norm_msg}")
                elif result is None and self.verbose >= 1:
                    tqdm.write(f"{prefix:<10} {colored(err_msg, COLOR.RED)}")
                self.progbar.update(1)

            # Collect the result and mark the task as done
            async with self.mainlk:
                self.shared_resources.append(
                    (True, result) if result is not None else (False, item)
                )
            self.queue.task_done()

    def run(self) -> tuple[list, list]:
        """Asynchronously (or sequentially) process the items.

        Returns
        -------
        results : list
            The successfully processed results by ``process_func``.
        failed_items : list
            The failed items.
        """
        self.shared_resources: list[tuple[bool, Any]] = []

        # Update the shared resources, either in synchronous or asynchronous mode
        if self._async_mode:
            asyncio.run(self._mainloop())
        else:
            all_items = list(self.items)
            print(f"Running in sequential mode for {len(all_items)} items...")
            self.progbar = tqdm(total=len(all_items))
            for item in all_items:
                result: Any
                norm_msg: str
                err_msg: str
                result, norm_msg, err_msg = asyncio.run(
                    self.process_func(item, **self.worker_kwargs[0])
                )
                if result is not None and self.verbose >= 2:
                    tqdm.write(norm_msg)
                elif result is None and self.verbose >= 1:
                    tqdm.write(colored(err_msg, COLOR.RED))
                self.shared_resources.append(
                    (True, result) if result is not None else (False, item)
                )
                self.progbar.update(1)
            self.progbar.close()

        # Collect finished results and failed items
        print("Collecting results...", end=" ", flush=True)
        results: list = []
        failed_items: list = []
        for passed, obj in self.shared_resources:
            if passed:
                results.append(obj)
            else:
                failed_items.append(obj)

        # Print a short summary of the execution
        print(
            f"Done/Failed - {colored(len(results), COLOR.GREEN)}/"
            f"{colored(len(failed_items), COLOR.RED)}"
        )
        if not failed_items:
            print(f"{EMOJI.STAR} All items have been successfully processed!")
        return results, failed_items
