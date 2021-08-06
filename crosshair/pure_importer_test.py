import importlib
import sys

from crosshair.pure_importer import prefer_pure_python_imports


def _unload_pydantic():
    for k in list(sys.modules):
        if k.startswith("pydantic"):
            del sys.modules[k]


def test_prefer_pure_python_imports():
    _unload_pydantic()
    pydantic = importlib.import_module("pydantic")
    assert pydantic.compiled

    _unload_pydantic()
    with prefer_pure_python_imports():
        pydantic = importlib.import_module("pydantic")
        assert not pydantic.compiled

    _unload_pydantic()
    pydantic = importlib.import_module("pydantic")
    assert pydantic.compiled
