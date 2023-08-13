"""This file is for asynchronous parallelization functionalities."""


from __future__ import annotations

import asyncio
from inspect import _ParameterKind, iscoroutinefunction, signature
from typing import Any, Callable, Iterable, NoReturn

from tqdm import tqdm

from ._color import COLOR, colored
from ._emoji import EMOJI
from .errors import InvalidParameterError


class AsyncRunner:
    """An asynchronous runner.

    Parameters
    ----------
    process_func : Callable, optional
        The processing function that takes a data item then returns three things: the
        result, a normal message, and an error message. If the processing succeeds, the
        error message should be ``None``. If the processing errors out, the result and
        the normal message should be ``None``.

        .. note::
            This function must be provided if the runner is intended to be run with a
            single worker.

        .. note::
            This function needs to accept variable-length keyword arguments. Also, this
            function is not supposed to raise any exception (though they will still be
            handled to avoid potential errors). Instead, all exceptions should be
            caught and reflected through the result, the normal message, and the error
            message as specified above.
    process_afunc : Callable, optional
        The asynchronous version of ``process_func``.

        .. note::
            This function must be provided if the runner is intended to be run with
            multiple workers.
        .. note::
            This function must be asynchronous. Also, in addition to accepting
            variable-length keyword arguments, it also needs to accept a keyword
            argument ``addtlks`` of type ``list[asyncio.Lock] | None`` with default
            value ``None``. It should also handle all possible exceptions, for the same
            reason as described in ``process_func``.
    verbose : int, default=1
        The verbosity level of the processing. For level 0, only a progress bar will be
        displayed. For level 1, the errored items will also be displayed. For levels
        higher than 2, all items will be displayed.
    """

    def __init__(
        self,
        process_func: Callable | None = None,
        process_afunc: Callable | None = None,
        verbose: int = 1,
    ):
        self.process_func = process_func
        self.process_afunc = process_afunc
        self.verbose = verbose

    def _validate_process_function(self, func: Callable, async_version: bool) -> None:
        """Validate the processing function.

        Parameters
        ----------
        func : Callable
            The function to evaluate.
        async_version : bool
            Whether the function to evaluate is the asynchronous processing function or
            the synchronous one.
        """
        if async_version ^ iscoroutinefunction(func):
            raise InvalidParameterError(
                "process_(a)func",
                actual=func,
                reason="must be (a)synchronous",
            )

        # Check the function signature
        sig = signature(func)
        if async_version and "addtlks" not in sig.parameters:
            raise InvalidParameterError(
                "process_afunc",
                actual=func,
                reason="must accept a keyword argument 'addtlks'",
            )
        elif async_version and sig.parameters["addtlks"].default is not None:
            raise InvalidParameterError(
                "process_afunc",
                actual=func,
                reason="keyword argument 'addtlks' must be defaulted to 'None'",
            )
        if not any(
            param.kind == _ParameterKind.VAR_KEYWORD
            for param in sig.parameters.values()
        ):
            raise InvalidParameterError(
                "process_(a)func",
                actual=func,
                reason="must accept variable-length **kwargs",
            )

    def _sequential(self) -> None:
        """Sequential processing."""
        all_items = list(self.items)
        self.progbar = tqdm(total=len(all_items))

        # Process each item sequentially
        for item in all_items:
            processed: Any | None = None
            unexpected_exception: str | None = None

            # Avoid explicitly raising the exception that interrupts the runner
            try:
                # Validated self.process_func is not None before calling
                processed = self.process_func(  # type: ignore[misc]
                    item, **self.worker_kwargs[0]
                )
            except Exception as e:
                unexpected_exception = f"UNEXPECTED {type(e).__name__}: {e!s:.30s}"
            if processed is not None:
                if not isinstance(processed, tuple):
                    unexpected_exception = (
                        "UNEXPECTED: 'process_func' returns a single value"
                    )
                elif len(processed) != 3:
                    unexpected_exception = (
                        f"UNEXPECTED: 'process_func' returns {len(processed)} values; "
                        "expected 3"
                    )

            # Get the processed information or treat the item as failed
            if unexpected_exception is not None:
                tqdm.write(colored(unexpected_exception, COLOR.RED))
                self.shared_resources.append((False, item))
                self.progbar.update(1)
                continue
            result, norm_msg, err_msg = processed

            # Print the execution information by demand
            if result is not None and self.verbose >= 2:
                tqdm.write(norm_msg)
            elif result is None and self.verbose >= 1:
                tqdm.write(colored(err_msg, COLOR.RED))
            self.shared_resources.append(
                (True, result) if result is not None else (False, item)
            )
            self.progbar.update(1)
        self.progbar.close()

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
        kwargs
            The additional keyword arguments.
        """
        while True:
            item = await self.queue.get()
            prefix = f"[W{worker_id:03d}]"
            processed: Any | None = None
            unexpected_exception: str | None = None

            # Avoid "task exception never being retrieved"
            try:
                # Validated self.process_afunc is not None before calling
                processed = await self.process_afunc(  # type: ignore[misc]
                    item, addtlks=self.addtlks, **kwargs
                )
            except Exception as e:
                unexpected_exception = f"UNEXPECTED {type(e).__name__}: {e!s:.30s}"
            if processed is not None:
                if not isinstance(processed, tuple):
                    unexpected_exception = (
                        "UNEXPECTED: 'process_afunc' returns a single value"
                    )
                elif len(processed) != 3:
                    unexpected_exception = (
                        f"UNEXPECTED: 'process_afunc' returns {len(processed)} values; "
                        "expected 3"
                    )

            # Get the processed information or treat the item as failed
            if unexpected_exception is not None:
                async with self.proglk:
                    tqdm.write(
                        f"{prefix:<10} {colored(unexpected_exception, COLOR.RED)}"
                    )
                    self.progbar.update(1)
                async with self.mainlk:
                    self.shared_resources.append((False, item))
                self.queue.task_done()
                continue
            result, norm_msg, err_msg = processed

            # Print the execution information by demand
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

    def run(
        self, items: Iterable, worker_kwargs: list[dict[str, Any]], n_locks: int = 0
    ) -> tuple[list, list]:
        """Asynchronously (or sequentially) process the items.

        Parameters
        ----------
        items : Iterable
            The items to process with either ``process_func`` or ``process_afunc``,
            depending on the number of workers.
        worker_kwargs : list of dict
            The additional keyword arguments to pass into ``process_func`` or
            ``process_afunc``. The length of ``worker_kwargs`` determines the number of
            workers to create, thus also determining whether to run in asynchronous or
            sequential mode.
        n_locks : int, default=0
            The additional locks to request. The requested locks will be passed to
            ``process_afunc`` via the keyword argument ``addtlks``. Not useful when
            running in sequential mode.

        Returns
        -------
        results : list
            The successfully processed results.
        failed_items : list
            The failed items.
        """
        self.items = items
        self.worker_kwargs = worker_kwargs
        self.n_locks = n_locks
        self.shared_resources: list[tuple[bool, Any]] = []

        # Update the shared resources, either in synchronous or asynchronous mode
        if len(self.worker_kwargs) > 1:
            if self.process_afunc is None:
                raise InvalidParameterError(
                    "process_afunc",
                    actual=self.process_afunc,
                    reason="must be defined when using multiple workers",
                )
            self._validate_process_function(self.process_afunc, async_version=True)
            asyncio.run(self._mainloop())
        else:
            if self.process_func is None:
                raise InvalidParameterError(
                    "process_func",
                    actual=self.process_func,
                    reason="must be defined when using a single worker",
                )
            self._validate_process_function(self.process_func, async_version=False)
            self._sequential()

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
