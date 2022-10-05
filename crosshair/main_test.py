import io
import re
import subprocess
import sys
import tempfile
from argparse import Namespace
from os.path import join, split

import pytest

from crosshair.fnutil import NotFound
from crosshair.main import (
    DEFAULT_OPTIONS,
    AnalysisKind,
    AnalysisMessage,
    AnalysisOptionSet,
    List,
    MessageType,
    Path,
    Tuple,
    check,
    describe_message,
    diffbehavior,
    set_debug,
    textwrap,
    unwalled_main,
    watch,
)
from crosshair.test_util import simplefs
from crosshair.util import add_to_pypath, load_file


@pytest.fixture(autouse=True)
def rewind_modules():
    defined_modules = list(sys.modules.keys())
    yield None
    for name, module in list(sys.modules.items()):
        # Some standard library modules aren't happy with getting reloaded.
        if name.startswith("multiprocessing"):
            continue
        if name not in defined_modules:
            del sys.modules[name]


@pytest.fixture
def root() -> Path:
    return Path(tempfile.mkdtemp())


def call_check(
    files: List[str], options: AnalysisOptionSet = AnalysisOptionSet()
) -> Tuple[int, List[str], List[str]]:
    stdbuf: io.StringIO = io.StringIO()
    errbuf: io.StringIO = io.StringIO()
    retcode = check(Namespace(target=files), options, stdbuf, errbuf)
    stdlines = [ls for ls in stdbuf.getvalue().split("\n") if ls]
    errlines = [ls for ls in errbuf.getvalue().split("\n") if ls]
    return retcode, stdlines, errlines


def call_diffbehavior(fn1: str, fn2: str) -> Tuple[int, List[str]]:
    buf: io.StringIO = io.StringIO()
    errbuf: io.StringIO = io.StringIO()
    retcode = diffbehavior(Namespace(fn1=fn1, fn2=fn2), DEFAULT_OPTIONS, buf, errbuf)
    lines = [
        ls for ls in buf.getvalue().split("\n") + errbuf.getvalue().split("\n") if ls
    ]
    return retcode, lines


SIMPLE_FOO = {
    "foo.py": """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return x + 1
"""
}

FOO_CLASS = {
    "foo.py": """
class Fooey:
  def incr(self, x: int) -> int:
    ''' post: _ == x '''
    return x + 1
"""
}
ASSERT_BASED_FOO = {
    "foo.py": """
def foofn(x: int) -> int:
  ''' :raises KeyError: when input is 999 '''
  assert x >= 100
  x = x + 1
  if x == 1000:
      raise KeyError
  assert x != 101
  return x
"""
}

FOO_WITH_CONFIRMABLE_AND_PRE_UNSAT = {
    "foo.py": """
def foo_confirmable(x: int) -> int:
  ''' post: _ > x '''
  return x + 1
def foo_pre_unsat(x: int) -> int:
  '''
  pre: x != x
  post: True
  '''
  return x
"""
}

OUTER_INNER = {
    "outer": {
        "__init__.py": "",
        "inner.py": """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return x
""",
    }
}

CIRCULAR_WITH_GUARD = {
    "first.py": """
import typing
if typing.TYPE_CHECKING:
    from second import Second
class First():
    def __init__(self, f: "Second") -> None:
        ''' post: True '''
""",
    "second.py": """
from first import First
class Second():
    pass
""",
}


DIRECTIVES_TREE = {
    "outerpkg": {
        "__init__.py": "# crosshair: off",
        "outermod.py": textwrap.dedent(
            """\
            def fn1():
                assert True
                raise Exception
            """
        ),
        "innerpkg": {
            "__init__.py": "# crosshair: on",
            "innermod.py": textwrap.dedent(
                """\
                # crosshair: off
                def fn2():
                    # crosshair: on
                    assert True
                    raise Exception  # this is the only function that's enabled
                def fn3():
                    assert True
                    raise Exception
                """
            ),
        },
    }
}


def test_load_file(root):
    simplefs(root, SIMPLE_FOO)
    module = load_file(join(root, "foo.py"))
    assert module is not None
    assert module.foofn(5) == 6


def test_check_by_filename(root):
    simplefs(root, SIMPLE_FOO)
    retcode, lines, _ = call_check([str(root / "foo.py")])
    assert retcode == 1
    assert len(lines) == 1
    assert "foo.py:3: error: false when calling foofn" in lines[0]


def test_check_by_class(root):
    simplefs(root, FOO_CLASS)
    with add_to_pypath(root):
        retcode, lines, _ = call_check(["foo.Fooey"])
        assert retcode == 1
        assert len(lines) == 1
        assert "foo.py:4: error: false when calling incr" in lines[0]


def test_check_failure_via_main(root):
    simplefs(root, SIMPLE_FOO)
    try:
        sys.stdout = io.StringIO()
        assert unwalled_main(["check", str(root / "foo.py")]) == 1
    finally:
        sys.stdout = sys.__stdout__


def test_check_ok_via_main(root):
    # contract is assert-based, but we do not analyze that type.
    simplefs(root, ASSERT_BASED_FOO)
    try:
        sys.stdout = io.StringIO()
        exitcode = unwalled_main(
            [
                "check",
                str(root / "foo.py"),
                "--analysis_kind=PEP316,icontract",
            ]
        )
        assert exitcode == 0
    finally:
        sys.stdout = sys.__stdout__


def test_no_args_prints_usage(root):
    try:
        sys.stderr = io.StringIO()
        exitcode = unwalled_main([])
    finally:
        out = sys.stderr.getvalue()
        sys.stderr = sys.__stderr__
    assert exitcode == 2
    assert re.search(r"^usage", out)


def DISABLE_TODO_test_assert_mode_e2e(root):
    simplefs(root, ASSERT_BASED_FOO)
    try:
        sys.stdout = io.StringIO()
        exitcode = unwalled_main(["check", root / "foo.py", "--analysis_kind=asserts"])
    finally:
        out = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
    assert exitcode == 1
    assert re.search(
        r"foo.py\:8\: error\: AssertionError\:  when calling foofn\(x \= 100\)", out
    )
    assert len([ls for ls in out.split("\n") if ls]) == 1


def test_directives(root):
    simplefs(root, DIRECTIVES_TREE)
    ret, out, err = call_check(
        [str(root)], AnalysisOptionSet(analysis_kind=[AnalysisKind.asserts])
    )
    assert err == []
    assert ret == 1
    assert re.search(r"innermod.py:5: error: Exception:  when calling fn2()", out[0])
    assert len(out) == 1


def test_directives_on_check_with_linenumbers(root):
    simplefs(root, DIRECTIVES_TREE)
    ret, out, err = call_check(
        [str(root / "outerpkg" / "innerpkg" / "innermod.py") + ":5"],
        AnalysisOptionSet(analysis_kind=[AnalysisKind.asserts]),
    )
    assert err == []
    assert ret == 1
    assert re.search(r"innermod.py:5: error: Exception:  when calling fn2()", out[0])
    assert len(out) == 1


def test_report_confirmation(root):
    simplefs(root, FOO_WITH_CONFIRMABLE_AND_PRE_UNSAT)
    retcode, lines, _ = call_check([str(root / "foo.py")])
    assert retcode == 0
    assert lines == []
    # Now, turn on confirmations with the `--report_all` option:
    retcode, lines, _ = call_check(
        [str(root / "foo.py")], options=AnalysisOptionSet(report_all=True)
    )
    assert retcode == 0
    assert len(lines) == 2
    output_text = "\n".join(lines)
    assert "foo.py:3: info: Confirmed over all paths." in output_text
    assert "foo.py:7: info: Unable to meet precondition." in output_text


def test_check_nonexistent_filename(root):
    simplefs(root, SIMPLE_FOO)
    retcode, _, errlines = call_check([str(root / "notexisting.py")])
    assert retcode == 2
    assert len(errlines) == 1
    assert "File not found" in errlines[0]
    assert "notexisting.py" in errlines[0]


def test_check_by_module(root):
    simplefs(root, SIMPLE_FOO)
    with add_to_pypath(root):
        retcode, lines, _ = call_check(["foo"])
        assert retcode == 1
        assert len(lines) == 1
        assert "foo.py:3: error: false when calling foofn" in lines[0]


def test_check_nonexistent_module(root):
    simplefs(root, SIMPLE_FOO)
    retcode, _, errlines = call_check(["notexisting"])
    assert retcode == 2
    assert (
        errlines[-1] == "crosshair.fnutil.NotFound: Module 'notexisting' was not found"
    )


def test_check_by_package(root):
    simplefs(root, OUTER_INNER)
    with add_to_pypath(root):
        retcode, lines, _ = call_check(["outer.inner.foofn"])
        assert retcode == 0
        assert len(lines) == 0


def test_check_nonexistent_member(root):
    simplefs(root, OUTER_INNER)
    with add_to_pypath(root), pytest.raises(NotFound):
        call_check(["outer.inner.nonexistent"])


def test_check_circular_with_guard(root):
    simplefs(root, CIRCULAR_WITH_GUARD)
    with add_to_pypath(root):
        retcode, lines, _ = call_check([str(root / "first.py")])
        assert retcode == 0


def test_watch(root):
    # Just to make sure nothing explodes
    simplefs(root, SIMPLE_FOO)
    retcode = watch(
        Namespace(directory=[str(root)]),
        AnalysisOptionSet(),
        max_watch_iterations=2,
    )
    assert retcode == 0


def test_diff_behavior_same(root):
    simplefs(root, SIMPLE_FOO)
    with add_to_pypath(root):
        retcode, lines = call_diffbehavior("foo.foofn", str(root / "foo.py:2"))
        assert lines == [
            "No differences found. (attempted 2 iterations)",
            "All paths exhausted, functions are likely the same!",
        ]
        assert retcode == 0


def test_diff_behavior_different(root):
    simplefs(
        root,
        {
            "foo.py": """
def add(x: int, y: int) -> int:
  return x + y
def faultyadd(x: int, y: int) -> int:
  return 42 if (x, y) == (10, 10) else x + y
"""
        },
    )
    with add_to_pypath(root):
        retcode, lines = call_diffbehavior("foo.add", "foo.faultyadd")
        assert retcode == 1
        assert lines == [
            "Given: (x=10, y=10),",
            "        foo.add : returns 20",
            "  foo.faultyadd : returns 42",
        ]


def test_diff_behavior_error(root):
    retcode, lines = call_diffbehavior("foo.unknown", "foo.unknown")
    assert retcode == 2
    assert re.search(".*NotFound", lines[0])


def test_diff_behavior_targeting_error(root):
    simplefs(root, SIMPLE_FOO)
    with add_to_pypath(root):
        retcode, lines = call_diffbehavior("foo.foofn", "foo")
        assert retcode == 2
        assert lines == ['"foo" does not target a function.']


def test_diff_behavior_via_main(root):
    simplefs(root, SIMPLE_FOO)
    sys.stdout = io.StringIO()
    try:
        with add_to_pypath(root):
            assert unwalled_main(["diffbehavior", "foo.foofn", "foo.foofn"]) == 0
    finally:
        out = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
    import re

    assert re.search("No differences found", out)


def test_cover(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    simplefs(tmp_path, SIMPLE_FOO)
    with add_to_pypath(str(tmp_path)):
        assert unwalled_main(["cover", "foo.foofn"]) == 0
    assert capsys.readouterr().out == "foofn(0)\n"


def test_main_as_subprocess(tmp_path: Path):
    # This helps check things like addaudithook() which we don't want to run inside
    # the testing process.
    simplefs(tmp_path, SIMPLE_FOO)
    completion = subprocess.run(
        ["python", "-m", "crosshair", "check", str(tmp_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert completion.returncode == 1
    assert "foo.py:3: error: false when calling foofn" in completion.stdout
    assert completion.stderr == ""


def test_mypycrosshair_command():
    example_file = join(
        split(__file__)[0], "examples", "icontract", "bugs_detected", "wrong_sign.py"
    )
    completion = subprocess.run(
        [
            f"python",
            f"-c",
            f"import crosshair.main;"
            + f"crosshair.main.mypy_and_check([r'{example_file}'])",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert completion.stderr.strip() == ""
    assert completion.returncode == 1


def test_describe_message():
    msg = AnalysisMessage(MessageType.PRE_UNSAT, "unsat", "filename", 1, 1, "traceback")
    opts = DEFAULT_OPTIONS.overlay(report_all=True, report_verbose=True)
    assert describe_message(msg, opts).split("\n") == [
        "traceback",
        "\x1b[91mI am having trouble finding any inputs that meet your preconditions.\x1b[0m",
        "filename:1:",
        "",
        "unsat",
        "",
    ]
