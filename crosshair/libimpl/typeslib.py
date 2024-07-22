from types import MappingProxyType
from typing import Any, Dict

from crosshair.core import register_patch, register_type


def _repr(map: MappingProxyType):
    if not isinstance(map, MappingProxyType):
        raise TypeError
    return f"mappingproxy({repr(dict(map))})"


def make_registrations():
    register_type(MappingProxyType, lambda p, kt=Any, vt=Any: MappingProxyType(p(Dict[kt, vt])))  # type: ignore
    register_patch(MappingProxyType.__repr__, _repr)
