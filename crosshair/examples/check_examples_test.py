"""Run functional tests of the tool on all the examples."""
import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pytest


def extract_linenums(text: str) -> List[int]:
    r"""
    Pull ordered line numbers out of crosshair output.

    >>> extract_linenums("foo:34:bar\nfoo:64:bar\n")
    [34, 64]
    """
    return list(map(int, re.compile(r":(\d+)\:").findall(text)))


def find_examples() -> Iterable[Path]:
    examples_dir = pathlib.Path(os.path.realpath(__file__)).parent
    for path in sorted(examples_dir.glob("**/*.py")):
        if path.stem != "__init__":
            yield path


def main(argv: List[str]) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", help="If set, overwrite the golden files", action="store_true"
    )
    args = parser.parse_args(argv)

    overwrite = bool(args.overwrite)

    success = True

    for pth in find_examples():
        success &= run_on_file(pth, overwrite)

    if not success:
        print("One or more functional tests failed. Please see above.")
        return 1

    print("The functional tests passed.")
    return 0


def run_on_file(pth: Path, overwrite: bool) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "crosshair",
        "check",
        str(pth),
    ]

    cmd_as_string = " ".join(shlex.quote(part) for part in cmd)

    print(f"Running: {cmd_as_string}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )

    stdout, stderr = proc.communicate()

    assert isinstance(stdout, str)
    assert isinstance(stderr, str)

    # We see empty output if, and only if, the process succeeds:
    if (proc.returncode == 0) != (stdout == "" and stderr == ""):
        print(
            f"The return code does not correspond to the output.\n\n"
            f"The command was:\n"
            f"{cmd_as_string}\n\n"
            f"The return code was: {proc.returncode}\n"
            f"The captured stdout was:\n"
            f"{stdout}\n\n"
            f"The captured stderr:\n"
            f"{stderr}\n\n"
        )
        return False

    expected_stdout_pth = pth.parent / (pth.stem + ".out")

    ##
    # Replace the absolute path to the examples directory
    # with a place holder to make these tests machine agnostic.
    ##

    path_re = re.compile(r"^.*[/\\]([_\w]+\.py):", re.MULTILINE)
    stdout, _ = path_re.subn(r"\1:", stdout)

    if overwrite:
        if expected_stdout_pth.exists():
            expected_stdout_pth.unlink()
        if stdout:
            expected_stdout_pth.write_text(stdout)
    else:
        if expected_stdout_pth.exists():
            expected_stdout = expected_stdout_pth.read_text()
        else:
            expected_stdout = ""

        # We only check line numbers, as error messages aren't stable.
        if extract_linenums(expected_stdout) != extract_linenums(stdout):
            print(
                f"The output was different than expected.\n\n"
                f"The command was:\n"
                f"{cmd_as_string}\n\n"
                f"The captured stdout was:\n"
                f"{stdout}\n\n"
                f"The expected stdout:\n"
                f"{expected_stdout}\n\n"
            )
            if stderr:
                print(f"The captured stderr was:\n" f"{stderr}\n\n")
            return False
    return True


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="only test 3rd party libs under new python versions",
)
@pytest.mark.parametrize("path", find_examples(), ids=lambda p: "_".join(p.parts[-3:]))
def test_examples(path: Path):
    # TODO: "unable to meet precondition" and non-deterministic problems aren't
    # surfaced. Reconsider.
    assert run_on_file(path, overwrite=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
