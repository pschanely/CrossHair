import importlib
import sys
from contextlib import contextmanager

import pytest

from crosshair.pure_importer import prefer_pure_python_imports


@contextmanager
def _pydantic_unloaded():
    saved = {}
    for k in list(sys.modules):
        if k.startswith("pydantic"):
            saved[k] = sys.modules.pop(k)
    try:
        yield
    finally:
        sys.modules.update(saved)


@pytest.mark.skipif(
    sys.version_info >= (3, 10), reason="no compiled pydantic on 3.10 (yet)"
)
def test_prefer_pure_python_imports():
    pydantic = importlib.import_module("pydantic")
    assert pydantic.compiled

    with _pydantic_unloaded():
        with prefer_pure_python_imports():
            pydantic = importlib.import_module("pydantic")
            assert not pydantic.compiled
