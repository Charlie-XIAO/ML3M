Installation
============

ml3m requires Python 3.10+ and can work on Linux/MacOS/Windows. Its dependencies include
the following:

- `numpy <https://numpy.org/>`_
- `openai <https://platform.openai.com/docs/api-reference/introduction?lang=python>`_
- `pandas <https://pandas.pydata.org/>`_
- `tqdm <https://tqdm.github.io/>`_

The recommended way to install ml3m is via pip:

.. code-block::

    pip install ml3m

After you have installed the package, you can use the following command to check the
versions and dependencies:

.. code-block::

    python -c "import ml3m; ml3m.show_versions()"

.. note::
    Older versions of Python (Python 3.9-) are not supported due to a known ``asyncio``
    bug which ml3m currently may invoke. We may try to support older versions in the
    future by using a different approach.
