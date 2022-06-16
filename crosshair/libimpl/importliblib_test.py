import subprocess

from crosshair.main import Path, textwrap
from crosshair.test_util import simplefs

DYNAMIC_IMPORT = {
    "__init__.py": "",
    "outer.py": textwrap.dedent(
        """\
        def outerfn(x: int) -> int:
            ''' post: _ == x '''
            from .innerx import innerfn
            return innerfn(x)
        """
    ),
    "innerx.py": textwrap.dedent(
        """
        from crosshair.tracers import is_tracing
        assert not is_tracing()
        def innerfn(x: int) -> int:
            return x
        """
    ),
}


def test_dynamic_import(tmp_path: Path):
    # This imports another module while checking.
    # The inner module asserts that tracing is not enabled.
    simplefs(tmp_path, DYNAMIC_IMPORT)
    ret = subprocess.run(
        ["python", "-m", "crosshair", "check", str(tmp_path / "outer.py")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert (ret.returncode, ret.stdout, ret.stderr) == (0, "", "")
