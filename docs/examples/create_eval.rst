Creating an Evaluation from Scratch
===================================

To create an evaluation from scratch, all one needs to prepare is the following:

- An original dataset, which includes all information needed for forming a query and
  the corresponding response.

The workflow is as follows:

- Use :class:`ml3m.base.ResponseGenerator` to generate the actual model responses,
  either appended to the original dataset or stored in another designated location.
  It requires a ``query`` parameter, which should be a function that takes a data item
  and outputs a model response.
- Extend :class:`ml3m.base.BaseEvaluator`, which requires overwriting
  :meth:`ml3m.base.BaseEvaluator._get_score` method. It helps scoring each data item
  in the dataset, and stores the scores as a ``.csv`` file.

There are currently three supported dataset formats: ``.jsonl``, ``.json``, and
``.csv``. These formats are loaded in different ways, and consequently the type of data
items may vary: 

- In a ``.jsonl`` dataset, each line will be loaded as a data item via
  :func:`json.loads`, either as a list or a dictionary.
  :meth:`ml3m.base.BaseEvaluator._get_score` method.
- In a ``.json`` dataset, the file content must be a json array of data. It will be
  loaded via :func:`json.load` as a list, and each item in the list will be treated as
  a data item, either being a list or a dictionary.
- In a ``.csv`` dataset, the file will be loaded by :func:`pandas.read_csv` into a
  :class:`pandas.DataFrame`. Each row, extracted via :meth:`pandas.DataFrame.iterrows`,
  will be treated as a data item of type :class:`pandas.Series`.

Step 1: Generate the responses to the data items
------------------------------------------------

The original datasets contain information about inputs and outputs, similar to training
sets. However, the outputs are *desired* outputs (or reference answers), and actual
model responses are also necessary for evalution. Therefore, the first step is to
append model responses to the corresponding data items.
:class:`ml3m.base.ResponseGenerator` serves this purpose.

Creating the ``query`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose that :download:`orig_dataset.jsonl <./files/orig_dataset.jsonl>` is the
original dataset. Each data item is a dictionary with keys "instruction", "input",
"output", and "history". The value of "history" is a list of input-output pairs used
multi-round conversations. We need to create a function that takes a data item as input
and returns the model response. For instance:

.. code-block:: python

    def query(data_item: dict) -> str:
        """Returns the model response."""
        query_list = []
        for conversation in data_item["history"]:
            query_list.append(f"### Human: {conversation[0]}")
            query_list.append(f"### Assistant: {conversation[1]}")
          query_list.append(f"### Human: {data_item['instruction']}")
        query = "\n".join(query_list)
        return model.chat(query)  # Replace with inference code

For different formats, the type for each data item may vary which has been previously
mentioned. This created function should always be consistent with the dataset format.
For instance, if the dataset is of ``.jsonl`` format but each data item is a list
instead of a dictionary, one should expect ``data_item`` to be a list. If the dataset
is of ``.csv`` format, one should expect ``data_item`` to be a :class:`pandas.Series`.


Creating the :class:`ml3m.base.ResponseGenerator` instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose that the aforementioned original dataset is placed at relative position
``datasets/orig_dataset.jsonl``. An example code can be as follows:

.. code-block:: python

    import os

    from ml3m.base import ResponseGenerator

    dirname = os.path.dirname(__file__)
    orig_dataset = os.path.join(dirname, "datasets", "orig_dataset.jsonl")
    dataset = os.path.join(dirname, "eval_datasets", "dataset.jsonl")

    generator = ResponseGenerator(
        orig_dataset=orig_dataset,
        dataset=dataset,
        query=query,  # The previously defined `query` function
        format="jsonl",  # This is by default
    )

With the above code, we will create a new dataset under relative position
``eval_datasets/dataset.jsonl`` that includes the model responses, with all information
in the original dataset preserved. Note that the created dataset would always be in the
same format as the original dataset.

There is also an option where no new dataset is created, but the model responses are
appended directly to the original dataset. To do so, simply specify ``dataset=None``.

Commonly this would not change the original dataset structure a lot. In ``.jsonl`` and
``.json`` formats when each data item is a dictionary, this simply adds a new key-value
pair. In ``.csv`` format, this adds a column to the loaded :class:`pandas.DataFrame`.
But it is worth mentioning that in ``.jsonl`` and ``.json`` formats when each data item
is a list, this will turn each data item into a dictionary with the key "data" pointing
to the original data item and a new key pointing to the generated model response. In
general, we would like to avoid having such dataset structures.

:class:`ml3m.base.ResponseGenerator` supports a few more options: 

- ``response_name``: The name of the new key/column. This is useful when the default
  name "response" already exists. Also, one may want to include responses of multiple
  models in the same file, and using different ``response_name`` for each model can
  serve the purpose.
- ``n_iter``: Models *can* fail. ``n_iter`` is the maximum number of iterations if the
  model fails on some data items.

For more information, please refer to its documentation.

Appending model responses to the original dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, it suffices to call the :meth:`ml3m.base.ResponseGenerator.generate` method.
