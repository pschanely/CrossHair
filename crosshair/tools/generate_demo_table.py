import importlib
import inspect
import itertools
import pkgutil
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote_plus

from crosshair.libimpl import builtinslib_test


def extract_webdemo(test_src):
    match = re.fullmatch(r".*?( *def f\(.*?)\s*check_states.*", test_src, re.DOTALL)
    if not match:
        print("*** Unable to find demo function in this test: ***")
        print(test_src)
        print("***  ***")
        sys.exit(1)
    (fnbody,) = match.groups()
    return "from typing import *\n\n" + textwrap.dedent(fnbody)


def extract_demo(module) -> list[tuple[str, str, str]]:
    ret = []
    for itemname in dir(module):
        item = getattr(module, itemname)
        if not hasattr(item, "pytestmark"):
            continue
        marks = [m for m in item.pytestmark if m.name == "demo"]
        if not marks:
            continue
        color = marks[0].args[0] if marks[0].args else "green"
        test_src = inspect.getsource(item)
        modname = itemname.removeprefix("test_")
        demo_src = extract_webdemo(test_src)
        ret.append((modname, color, demo_src))
    return sorted(ret)


def stdlib_demos() -> dict[str, list[tuple[str, str, str]]]:
    libimpl_dir = Path(__file__).parent.parent / "libimpl"
    ret = {}
    for (_f, name, _p) in pkgutil.iter_modules([str(libimpl_dir)]):
        if not name.endswith("lib_test"):
            continue
        if name == "builtinslib_test":
            continue
        modname = f"crosshair.libimpl.{name}"
        mod = importlib.import_module(modname)
        ret[name.removesuffix("lib_test")] = extract_demo(mod)
    return ret


def format_line(line: list[tuple[str, str, str]]) -> str:
    return " ".join(f'[{char}]({url} "{name}")' for (char, name, url) in line)


def markdown_table(sections: dict[str, dict[str, list[tuple[str, str, str]]]]) -> str:
    # table_width = max(len(line) for section in sections.values() for line in section.values())
    parts = ["||||\n|-|-|-|\n"]
    for section_name, section in sections.items():
        section = section.copy()
        parts.append(f"|{section_name}||{format_line(section.pop('', []))}|\n")
        for line_header, line in section.items():
            parts.append(f"||{line_header}|{format_line(line)}|\n")
    return "".join(parts)


COLORS = {
    "yellow": "&#x1F7E1;",
    "green": "&#x1F7E2;",
    "red": "&#x1F534;",
}

CH_WEB = "https://crosshair-web.org"

SPECIAL_CHARS = {
    "__eq__": "==",
    "__le__": "<=",
    "__contains__": "in",
    "__getitem__": "[]",
    "__setitem__": "[]=",
    "__floordiv__": "//",
    "__truediv__": "/",
    "__mod__": "%",
    "__add__": "+",
    "__mul__": "*",
    "__pow__": "**",
    "__sub__": "-",
}


def make_link(name: str, color: str, src: str) -> tuple[str, str, str]:
    char = SPECIAL_CHARS.get(name, name.removeprefix("__")[0])
    text = f"<b>{char}</b><sup>{COLORS[color]}</sup>"
    return (text, name, f"{CH_WEB}/?source={quote_plus(src)}")


builtins_demos = extract_demo(builtinslib_test)
builtins_section: dict[str, list[tuple[str, str, str]]] = {}
for key, group in itertools.groupby(
    builtins_demos, key=lambda p: p[0].split("_", 1)[0]
):
    builtins_section[key] = [
        make_link(name.split("_", 1)[1], color, src) for (name, color, src) in group
    ]


def divide_stdlib_module(
    modulename: str, items: list[tuple[str, str, str]]
) -> dict[str, list[tuple[str, str, str]]]:
    ret = defaultdict(list)
    for (name, color, src) in items:
        if name.endswith("_method"):
            name = name.removesuffix("_method")
            (classname, methodname) = name.split("_", 1)
        else:
            (classname, methodname) = ("", name)
        ret[classname].append(
            make_link(methodname, color, f"import {modulename}\n" + src)
        )
    return ret


stdlib = {}
for (modulename, items) in stdlib_demos().items():
    stdlib[modulename] = divide_stdlib_module(modulename, items)


print(markdown_table({"builtins": builtins_section, **stdlib}))
