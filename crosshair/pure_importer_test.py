import importlib
import sys
from contextlib import contextmanager

import pytest

from crosshair.pure_importer import prefer_pure_python_imports


@pytest.mark.skip(
    reason="We used to test pydantic here, but current version doesn't use Cython"
)
def test_prefer_pure_python_imports():
    with prefer_pure_python_imports():
        pydantic = importlib.import_module("pydantic")
        assert not pydantic.compiled
