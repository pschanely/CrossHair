import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import FileFinder
from typing import Any, Callable


@dataclass
class PreferPureLoaderHook:
    orig_hook: Callable[[str], Any]

    def __call__(self, path: str):
        finder = self.orig_hook(path)
        if isinstance(finder, FileFinder):
            # Move pure python file loaders to the front
            finder._loaders.sort(key=lambda pair: 0 if pair[0] in (".py", ".pyc") else 1)  # type: ignore
        return finder


@contextmanager
def prefer_pure_python_imports():
    sys.path_hooks = [PreferPureLoaderHook(h) for h in sys.path_hooks]
    sys.path_importer_cache.clear()
    yield
    assert all(isinstance(h, PreferPureLoaderHook) for h in sys.path_hooks)
    sys.path_hooks = [h.orig_hook for h in sys.path_hooks]
    sys.path_importer_cache.clear()
