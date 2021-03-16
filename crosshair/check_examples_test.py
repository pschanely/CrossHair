"""Run functional tests of the tool on all the examples."""
import argparse
import fnmatch
import os
import pathlib
import re
import shlex
import subprocess
import sys
from typing import List


def extract_linenums(text: str) -> List[int]:
    r"""
    Pull ordered line numbers out of crosshair output.

    >>> extract_linenums("foo:34:bar\nfoo:64:bar\n")
    [34, 64]
    """
    return list(map(int, re.compile(r":(\d+)\:").findall(text)))


def main(argv: List[str]) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", help="If set, overwrite the golden files", action="store_true"
    )
    parser.add_argument(
        "--include",
        help="Run functional tests only on the given files; "
        "glob patterns also possible "
        "(example: '**/icontract/**')",
        nargs="+",
    )
    parser.add_argument(
        "--exclude",
        help="Exclude functional tests for the given files; "
        "glob patterns also possible "
        "(example: '**/showcase.py')",
        nargs="+",
    )
    parser.add_argument(
        "--continue_on_error",
        help="If set, continue the remainder of the tests despite errors",
        action="store_true",
    )
    args = parser.parse_args(argv)

    overwrite = bool(args.overwrite)

    include_pths = (
        set(pathlib.Path(pth) for pth in args.include)
        if args.include is not None
        else set()
    )

    exclude_pths = (
        set(pathlib.Path(pth) for pth in args.exclude)
        if args.exclude is not None
        else set()
    )

    continue_on_error = bool(args.continue_on_error)

    this_path = pathlib.Path(os.path.realpath(__file__))
    cwd = pathlib.Path(os.getcwd())

    # We need to append CWD to the relative include and exclude paths
    # so that they can be used in set and match operations.
    #
    # We can not use ``.resolve()`` since we allow glob patterns.

    include_pths = set(pth if pth.is_absolute() else cwd / pth for pth in include_pths)

    exclude_pths = set(pth if pth.is_absolute() else cwd / pth for pth in exclude_pths)

    def strip_cwd(path: pathlib.Path) -> pathlib.Path:
        """Remove cwd from the path, if prefixed accordingly."""
        if cwd in path.parents:
            return path.relative_to(cwd)

        return path

    ##
    # Run tests
    ##

    examples_dir = this_path.parent / "examples"

    success = True

    for kind in ["PEP316", "icontract"]:
        for outcome in ["correct_code", "bugs_detected"]:
            for pth in sorted((examples_dir / kind / outcome).glob("*.py")):
                if pth.stem == "__init__":
                    continue

                # We don't use ``.resolve()`` here as this might confuse
                # the user (*e.g.*, when symbolic links are automatically
                # resolved so that some paths are unexpectedly filtered in or
                # out).
                pth_abs = pth if pth.is_absolute() else cwd / pth

                if len(include_pths) > 0 and (
                    pth_abs not in include_pths
                    and not any(
                        fnmatch.fnmatch(str(pth_abs), str(pattern))
                        for pattern in include_pths
                    )
                ):
                    continue

                if len(exclude_pths) > 0 and (
                    pth_abs in exclude_pths
                    or any(
                        fnmatch.fnmatch(str(pth_abs), str(pattern))
                        for pattern in exclude_pths
                    )
                ):
                    continue

                cmd = [
                    sys.executable,
                    "-m",
                    "crosshair",
                    "check",
                    str(strip_cwd(pth)),
                    "--analysis_kind",
                    kind,
                    "--per_condition_timeout=0.75",
                ]

                cmd_as_string = " ".join(shlex.quote(part) for part in cmd)

                print(f"Running: {cmd_as_string}")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                )

                stdout, stderr = proc.communicate()

                assert isinstance(stdout, str)
                assert isinstance(stderr, str)

                if outcome == "correct_code":
                    if proc.returncode != 0:
                        print(
                            f"The functional test failed on an example. "
                            f"Expected a success (a return code 0), "
                            f"but got a return code {proc.returncode}.\n\n"
                            f"The stdout was:\n{stdout}\n\n"
                            f"The stderr was:\n{stderr}\n\n"
                            f"The command was:\n{cmd_as_string}",
                            file=sys.stderr,
                        )

                        if continue_on_error:
                            success = False
                            continue
                        else:
                            return -1

                elif outcome.startswith("bugs_detected"):
                    if proc.returncode != 1:
                        print(
                            f"The functional test failed on an example. "
                            f"Expected a failure (a return code equal to 1), "
                            f"but got a return code {proc.returncode}.\n\n"
                            f"The stdout was:\n{stdout}\n\n"
                            f"The stderr was:\n{stderr}\n\n"
                            f"The command was:\n{cmd_as_string}",
                            file=sys.stderr,
                        )

                        if continue_on_error:
                            success = False
                            continue
                        else:
                            return -1
                else:
                    raise AssertionError(f"Unexpected outcome: {outcome!r}")

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
                            f"The captured output does not correspond to "
                            f"the expected output.\n\n"
                            f"The command was:\n"
                            f"{cmd_as_string}\n\n"
                            f"The captured stdout was:\n"
                            f"{stdout}\n\n"
                            f"The expected stdout:\n"
                            f"{expected_stdout}\n\n"
                        )
                        if continue_on_error:
                            success = False
                            continue
                        else:
                            return -1

    if not success:
        print("One or more functional tests failed. Please see above.")
        return -1

    print("The functional tests passed.")
    return 0


def test_examples():
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
