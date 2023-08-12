import os
import random
import shutil

import pytest


@pytest.fixture(scope="module")
def storage(request):
    """Make a temporary storage and clear it towards the end."""
    dirname = os.path.dirname(__file__)
    storage = os.path.join(dirname, str(random.randint(1000000, 9999999)))
    while os.path.exists(storage):
        storage = os.path.join(
            dirname, str(random.randint(1000000, 9999999))
        )  # pragma: no cover
    os.mkdir(storage)
    with open(os.path.join(storage, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("*")

    def cleanup():
        shutil.rmtree(storage)

    request.addfinalizer(cleanup)
    return storage
