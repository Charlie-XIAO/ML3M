import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Callable, Generator

import pandas as pd
from tqdm import tqdm

from .._logging import manage_timed_logs
from .._typing import DataItemType, DatasetFormat


class ResponseGenerator:
    """Generate response and combine with the original dataset.

    To use this generator, an original dataset of a set of inputs and expected
    responses needs to prepared. In other words, it is similar to a training set.
    The original dataset can take the following formats:

    - ``.jsonl``: Each line will be read by :func:`json.loads`, either as a list or as
      a dictionary. This is to be handled by ``query``.

    - ``.json``: The whole file will be read by :func:`json.load`, and it should be a
      json array (and read as a list). Each item in the list will be treated in the
      same way as each line in the ``.jsonl`` format. It can be either a list or a
      dictionary, to be handled by ``query``.

    - ``.csv``: The whole file will be read by :func:`pandas.read_csv` as a
      :class:`pandas.DataFrame`. Each row is to be handled by ``query``.

    The resulting dataset expand the original dataset by including the actual model
    responses. It will be in the same format as the original dataset. The rules are as
    follows:

    - ``.jsonl``: If each line is read as a list, this will turn that line into a
      dictionary with the key "data" pointing to that list and the key specified by
      ``response_name`` pointing to the corresponding response. If each line is read as
      a dictionary, there will be an additional key specified by ``response_name`` with
      value as the corresponding response. If that key already exists, an exception
      will be raised.

    - ``.json``: This follows the same rules as ``.jsonl``, with each line being each
      item in the json array here.

    - ``.csv``: This will add a column with its name specified by ``response_name``. If
      that column name already exists, an exception will be raised.

    Parameters
    ----------
    orig_dataset : str or pathlib.Path
        The absolute path to the original dataset.
    dataset : str or pathlib.Path or None
        The absolute path to the result dataset. If ``None``, ``orig_dataset`` will
        be overwritten (but no original data will be lost; there will only be
        additional information added).
    query : Callable
        The function that queries a model given a data item and outputs the model
        response. The input parameter should be a :class:`pandas.Series`, a list, or a
        dictionary, depending on ``format``. The output should be a single string
        representing the model response.
    format : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``, as specified above.
    response_name : str, default="response"
        The key or column name to use for the response.
    n_iter : int, default=3
        The maximum number of iterations if OpenAI querying failed on any data item.
    verbose : int, default=1
        The verbosity level of printout. For level 0, only a progress bar will be
        displayed. For level 1, the errored queries will also be displayed. For level
        higher than 2, all queries will be displayed. Regardless of the verbosity
        level, the full log will be written, except that the verbosity of exceptions
        will depend on ``err_verbose``.
    err_verbose : int, default=1
        The verbosity level of the error message when writing logs. For level 0, only
        the exception type will be included. For level 1, the exception message will
        also be included. For level higher than 2, the full stack trace will be
        included. Regardless of the ``err_verbose``, verbosity level 0 will be used in
        printout of error messages.
    """

    def __init__(
        self,
        orig_dataset: str | Path,
        dataset: str | Path | None,
        query: Callable[[DataItemType], str],
        *,
        format: DatasetFormat = "jsonl",
        response_name: str = "response",
        n_iter: int = 3,
        verbose: int = 1,
        err_verbose: int = 1,
    ) -> None:
        self.orig_dataset = orig_dataset
        self.dataset = dataset
        self.query = query
        self.format = format
        self.response_name = response_name
        self.n_iter = n_iter
        self.verbose = verbose
        self.err_verbose = err_verbose

        # Validate the paths and format
        if not os.path.exists(self.orig_dataset):
            raise ValueError(
                f"Original dataset not found at {os.path.abspath(self.orig_dataset)}"
            )
        if self.format not in ["jsonl", "json", "csv"]:
            raise ValueError(f"Invalid format: {self.format}")

    def _check_item(self, data: list):
        """Check if each item of data is a list or dictionary.

        Parameters
        ----------
        data : list
            The list of data items.

        Returns
        -------
        islist : bool
            Whether the data items are lists.
        """
        if all(isinstance(item, list) for item in data):
            return True
        elif all(isinstance(item, dict) for item in data):
            return False
        else:
            raise ValueError("Data has internally-inconsistent types.")

    def generate(self, *, overwrite: bool = False) -> bool:
        """Generate response and combine with the original dataset.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the responses if some already exist. If
            ``format="jsonl"`` or ``format="json"``, this will look for the
            key specified by ``response_name`` in each item. If ``format="csv"``,
            this will look for the column specified by ``response_name``.

        Returns
        -------
        completed : bool
            Whether the task has been fully completed.
        """
        completed: bool = False
        if self.format not in ["jsonl", "json", "csv"]:
            raise ValueError(f"Invalid format: {self.format}")

        # Validate the destination path
        manual_overwrite: bool = False
        if self.dataset is not None and not os.path.exists(self.dataset):
            abs_save_path = os.path.abspath(self.dataset)
            print(
                f"\033[93mSave path not found at {abs_save_path}; forcefully created"
                "\033[0m",
            )
            directories, _ = os.path.split(abs_save_path)
            if not os.path.exists(directories):
                os.makedirs(directories)
            shutil.copyfile(self.orig_dataset, self.dataset)
            manual_overwrite = True
        destination = self.orig_dataset if self.dataset is None else self.dataset
        mlog_path = manage_timed_logs(prefix="model", keep=10)

        for it in range(self.n_iter):
            print(f"### Iteration {it}")
            # Load the data and create the data iterator
            data: list | pd.DataFrame
            islist: bool = False
            n_tot: int
            datait: Generator[tuple[int, DataItemType], None, None]
            if self.format in ["jsonl", "json"]:
                with open(destination, "r", encoding="utf-8") as f:
                    if self.format == "jsonl":
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                islist = self._check_item(data)
                if it != 0 or not (overwrite or manual_overwrite):
                    if islist:
                        raise ValueError(
                            "``overwrite=False`` is not possible if data items are "
                            "lists since it is vague where to look for the existing "
                            "response."
                        )
                    elif any(self.response_name not in item for _, item in data):
                        raise ValueError(
                            "Some data items are missing the key "
                            f"'{self.response_name}'"
                        )
                    else:
                        datait = (
                            (i, item)
                            for i, item in enumerate(data)
                            if pd.isna(item[self.response_name])
                        )
                        n_tot = sum(pd.isna(item[self.response_name] for item in data))
                else:
                    datait = ((i, item) for i, item in enumerate(data))
                    n_tot = len(data)
            elif self.format == "csv":
                data = pd.read_csv(destination)
                if it != 0 or not (overwrite or manual_overwrite):
                    if self.response_name not in data.columns:
                        raise ValueError(f"Missing column '{self.response_name}'")
                    filtered_data = data[pd.isna(data[self.response_name])]
                    datait = ((i, row) for i, row in filtered_data.iterrows())
                    n_tot = len(filtered_data)
                else:
                    datait = ((i, row) for i, row in data.iterrows())
                    n_tot = len(data)

            # Query each data item and update loaded data in-place
            progbar, n_failed = tqdm(total=n_tot), 0
            for i, data_item in datait:
                response: str | None = None
                errmsg: str | None = None
                pmsg: str | None = None
                try:
                    response = self.query(data_item)
                except Exception as e:
                    if self.err_verbose >= 2:
                        errmsg = traceback.format_exc()
                    elif self.err_verbose == 1:
                        errmsg = f"{type(e).__name__}: {e}"
                    else:
                        errmsg = type(e).__name__
                    pmsg = type(e).__name__
                    n_failed += 1

                # Update the loaded data based on its format
                if self.format in ["jsonl", "json"]:
                    if islist:
                        data[i] = {"data": data_item, self.response_name: response}
                    elif self.response_name in data[i]:
                        raise ValueError(f"{self.response_name} already exists")
                    else:
                        data[i][self.response_name] = response
                elif self.format == "csv":
                    data.at[i, self.response_name] = response

                # Write the log and print information based on verbosity level
                mlog_item = {
                    "id": i,
                    "kwds": {
                        "orig_dataset": self.orig_dataset,
                        "dataset": self.dataset,
                    },
                    "response": response,
                    "errmsg": errmsg,
                }
                with open(mlog_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(mlog_item, ensure_ascii=False))

                # Printing information based on the verbosity level
                prefix = f"[Index.{i}, It.{it}]"
                if errmsg is None and self.verbose >= 2:
                    tqdm.write(f"{prefix:<30} {response:.30s}...")
                elif errmsg is not None and self.verbose >= 1:
                    tqdm.write(f"{prefix:<30} \033[31m{pmsg}\033[0m")
                progbar.update(1)

            # Write the data (may be temporary)
            progbar.close()
            print("Writing data...", end=" ", flush=True)
            if self.format in ["jsonl", "json"]:
                with open(destination, "w", encoding="utf-8"):
                    if self.format == "jsonl":
                        for data_item in data:
                            f.write(json.dumps(data_item, ensure_ascii=False))
                    elif self.format == "json":
                        json.dump(data, f, ensure_ascii=False, indent=4)
            elif self.format == "csv":
                data.to_csv(destination, index=False)
            print(f"done, available at:\n{os.path.abspath(destination)}")

            # Print execution summary
            if n_failed == 0:
                print("\033[92mAll items done.\033[0m")
                completed = True
                break
            elif it != self.n_iter - 1:
                manual_overwrite = False
                print(f"\033[31m{n_failed} items failed, reiterating...\033[0m")
        return completed
