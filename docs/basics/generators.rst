Generating Model Responses
==========================

The first step to evaluate the performance of an LLM is to generate its responses to
the evaluation datasets.

Commonly evaluation datasets are in the same form as training datasets, in which each
data item contains an input and an output (or information sufficient to construct the
input and the output). This output, in an evaluation dataset, can be treated as *ground
truth* or *reference output*, which is what we except our LLM to respond based on the
corresponding input.

ml3m provides the functionality to generate model responses and append them to the
original evaluation dataset. The relevant class is:

.. currentmodule:: ml3m

.. autosummary::
    :nosignatures:

    base.ResponseGenerator

To use :class:`ml3m.base.ResponseGenerator`, you would need to prepare an original
evaluation dataset ``orig_dataset`` and a saving location ``dataset`` to for storing
the original data *and* the responses. The reason for not updating ``orig_dataset`` in
place is to keep the original evaluation dataset clean for other possible use. There
are currently three supported formats, as will be introduced next.

.. _Dataset Format:

Dataset Formats and Data Items
------------------------------

For now, let us suppose ``response_name="my_response"``, which is a required parameter
to the :class:`ml3m.base.ResponseGenerator` specifying the key/column name of the
response. However, in practice you should be extremely careful when picking
``response_name``; it should not be a key/column name that exists in ``orig_dataset``.

.. raw:: html

    <details>
    <summary><code>jsonl</code></summary>
    <p></p>

``jsonl`` is the default dataset format. As follows is a simple example:

.. code-block::

    {"instruction": "What is the capital of China?", "output": "Beijing."}
    {"instruction": "What is the capital of France?", "output": "Paris."}

In this case, each line of the dataset is considered as a data item, loaded as a
dictionary, e.g., ``{"instruction": "What is the capital of China?", "output":
"Beijing."}``. The resulting dataset will be of the same format, which looks like the
following:

.. code-block::

    {"instruction": "What is the capital of China?", "output": "Beijing.", "my_response": "xxx"}
    {"instruction": "What is the capital of France?", "output": "Paris.", "my_response": "xxx"}

One other possible example in the ``jsonl`` format is:

.. code-block::

    ["What is the capital of China?", "Beijing."]
    ["What is the capital of France?", "Paris."]

In this case, each data item will be loaded as a list, e.g.,
``["What is the capital of China?", "Beijing."]``, and the resulting dataset will be in
the following form:

.. code-block::

    {"data": ["What is the capital of China?", "Beijing."], "my_response": "xxx"}
    {"data": ["What is the capital of France?", "Paris."], "my_response": "xxx"}

However, this second example is *not recommended*.

.. raw:: html

    </details>
    <p></p>

    <details>
    <summary><code>json</code></summary>
    <p></p>

The ``json`` format needs to be specified by ``fmt="json"``. As follows is a simple
example:

.. code-block::

    [
        {
            "instruction": "What is the capital of China?",
            "output": "Beijing."
        },
        {
            "instruction": "What is the capital of France?",
            "output": "Paris."
        }
    ]

The overall dataset *must* be loaded as a JSON array, where each object in that array
will be considered as a data item, e.g., ``{"instruction": "What is the capital of
China?", "output": "Beijing."}`` of type :class:`dict`. The resulting dataset will be
of the same format, which looks like the following:

.. code-block::

    [
        {
            "instruction": "What is the capital of China?",
            "output": "Beijing.",
            "my_response": "xxx"
        },
        {
            "instruction": "What is the capital of France?",
            "output": "Paris.",
            "my_response": "xxx"
        }
    ]

One other possible example in the ``json`` format is:

.. code-block::

    [
        [
            "What is the capital of China?",
            "Beijing."
        ],
        [
            "What is the capital of France?",
            "Paris."
        ]
    ]

In this case, each data item will be loaded as a list, e.g.,
``["What is the capital of China?", "Beijing."]``, and the resulting dataset will be in
the following form:

.. code-block::

    [
        {
            "data": [
                "What is the capital of China?",
                "Beijing."
            ],
            "my_response": "xxx"
        },
        {
            "data": [
                "What is the capital of France?",
                "Paris."
            ],
            "my_response": "xxx"
        }
    ]

However, this second example is *not recommended*.

.. raw:: html

    </details>
    <p></p>

    <details>
    <summary><code>csv</code></summary>
    <p></p>

The ``csv`` format needs to be specified by ``fmt="csv"``. As follows is a simple
example:

.. code-block::

    instruction,output
    What is the capital of China?,Beijing.
    What is the capital of France?,Paris.

The dataset will be loaded as a :class:`pandas.DataFrame`, where each row will be
considered as a data item, loaded as a :class:`pandas.Series`. The resulting dataset
will be of the same format, which looks like the following:

.. code-block::

    instruction,output,my_response
    What is the capital of China?,Beijing.,xxx
    What is the capital of France?,Paris.,xxx

One other possible example in the ``csv`` format is:

.. code-block::

    [
        [
            "What is the capital of China?",
            "Beijing."
        ],
        [
            "What is the capital of France?",
            "Paris."
        ]
    ]

.. raw:: html

    </details>
    <p></p>

Defining the Querying Function
------------------------------

In addition to the original evaluation dataset ``orig_dataset``, the saving location
``dataset``, and the key/column name ``response_name``,
:class:`ml3m.base.ResponseGenerator` further requires a parameter ``query_func``, which
should be a function that accepts a data item and returns the response of the model.

The type of the data items can vary based on ``fmt``, which has been specified in the
previous subsection. Therefore, ``query_func`` must be corresponding to ``fmt``. Its
return value should be *only* a string representing the model response.

There are a few important points worth noting:

- ``query_func`` should not catch any exception that causes the data item to fail,
  otherwise the :class:`ml3m.base.ResponseGenerator` will not be able to notice that
  the data item has failed (unless this is intentional).

- ``query_func`` can be defined either as synchronous or as asynchounous. If it is
  defined as synchrounous, you must specify ``n_workers=1``, and otherwise
  ``n_workers>1``. Defining an asynchronous ``query_func`` is useful when your model
  can be parallelized when performing inference, which can significantly improve the
  speed. If your model does not support parallelization, making it asynchounous will be
  meaningless.

- ``query_func`` should not contain model initialization code but *only* model
  inference code, since ``query_func`` is executed in a loop.

Generating the Responses
------------------------

Here is a code snippet using :class:`ml3m.base.ResponseGenerator` to generate the model
responses. It takes the first example in the ``jsonl`` format in
:ref:`this section <Dataset Format>`.

.. code-block:: python

    import os

    import torch
    from ml3m.base import ResponseGenerator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.utils import GenerationConfig

    # Obtain the dataset paths, assuming ``orig_dataset.jsonl`` is the original
    # evaluation dataset (existent) and ``dataset.jsonl`` is the desired saving
    # location (may or may not exist)
    # You should use your own (absolute) paths

    dirname = os.path.dirname(__file__)
    orig_dataset = os.path.join(dirname, "orig_dataset.jsonl")
    dataset = os.path.join(dirname, "dataset.jsonl")

    # The initialization should be done in advance
    # You should use your own initialization code (if any)

    tokenizer = AutoTokenizer.from_pretrained("xxx", use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("xxx", device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)

    # Define the querying function (in this example, data_item is a dictionary)
    # You should define your own querying function based on your dataset structure and
    # use your own model inference code

    def query_func(data_item):
        question = data_item["instruction"]
        response = model.chat(tokenizer,
                              [{"role": "user", "content": question}])  # Inference
        return response

    # Create the generator and generate the responses

    generator = ResponseGenerator(
        orig_dataset=orig_dataset,
        dataset=dataset,
        query_func=query_func,
        response_name="my_response",
        fmt="jsonl",  # default
        n_workers=1,  # default; query_func is synchronous
        logging_mode="all",  # default
        verbose=1,  # default
    )
    generator.generate()

The model *may* fail certain data items, and :class:`ml3m.base.ResponseGenerator` takes
this into account. :meth:`ml3m.base.ResponseGenerator.generate` returns a boolean value
to indicate whether any data item has errored out. Moreover, by default it will not
regenerate responses for the data items that already have corresponding responses in
the ``dataset``. Therefore, it is safe to either execute this code multiple times, or
do something like:

.. code-block:: python

    max_iter = 5
    for _ in range(max_iter):
        completed = generator.generate()
        if completed:
            break

:meth:`ml3m.base.ResponseGenerator.generate` also provides a keyword parameter
``overwrite`` that ignores the existing responses in the ``dataset``, and one should be
careful setting it to ``True``.

Generating the Responses of Multiple Models
-------------------------------------------

It is common that we need to compare the performances of multiple models. In this case,
it would be nice to generate the responses of multiple models to ``orig_dataset`` to
the same ``dataset``. :class:`ml3m.base.ResponseGenerator` serves this purpose simply
by specifying different ``response_name``. For instance:

.. code-block:: python

    from functools import partial

    def query_func(data_item, model, tokenizer):
        question = data_item["instruction"]
        response = model.chat(tokenizer, [{"role": "user", "content": question}])
        return response

    generators = [
        ResponseGenerator(
            orig_dataset=orig_dataset,
            dataset=dataset,
            query_func=partial(query_func, model=model, tokenizer=tokenizer),
            response_name="model1_response",
        )
        for model, tokenizer in zip([model1, model2], [tokenizer1, tokenizer2])
    ]

    for generator in generators:
        generator.generate()
