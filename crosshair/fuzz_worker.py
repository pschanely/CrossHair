import importlib
import sys

import afl  # type: ignore

from .base import get_conditions
from .util import walk_qualname


def main():
    package, module_name, *fn_names = sys.argv[1:]
    module = importlib.import_module(module_name, package=package)
    fns = [walk_qualname(module, name) for name in fn_names]
    conditions = []
    for fn in fns:
        conditions.extend(get_conditions(fn))
    while afl.loop(200):
        buf = memoryview(bytearray(sys.stdin.buffer.read()))
        for condition in conditions:
            status, _, _ = condition.check_buffer(buf)


if __name__ == '__main__':
    main()
