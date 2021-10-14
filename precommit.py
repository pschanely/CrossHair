#!/usr/bin/env python3
"""Run pre-commit checks on the repository."""
import argparse
import enum
import os
import pathlib
import subprocess
import sys


class Step(enum.Enum):
    BLACK = "black"
    FLAKE8 = "flake8"
    ISORT = "isort"
    PYDOCSTYLE = "pydocstyle"
    MYPY = "mypy"
    DOCTEST = "doctest"
    CHECK_INIT_AND_SETUP_COINCIDE = "check-init-and-setup-coincide"
    CHECK_HELP_IN_DOC = "check-help-in-doc"
    TEST = "test"


def main() -> int:
    """"Execute entry_point routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        help="Try to automatically fix the offending files (e.g., by re-formatting).",
        action="store_true",
    )
    parser.add_argument(
        "--select",
        help=(
            "If set, only the selected steps are executed. "
            "This is practical if some of the steps failed and you want to "
            "fix them in isolation. "
            "The steps are given as a space-separated list of: "
            + " ".join(value.value for value in Step)
        ),
        metavar="",
        nargs="+",
        choices=[value.value for value in Step],
    )
    parser.add_argument(
        "--skip",
        help=(
            "If set, skips the specified steps. "
            "This is practical if some of the steps passed and "
            "you want to fix the remainder in isolation. "
            "The steps are given as a space-separated list of: "
            + " ".join(value.value for value in Step)
        ),
        metavar="",
        nargs="+",
        choices=[value.value for value in Step],
    )

    args = parser.parse_args()

    overwrite = bool(args.overwrite)

    selects = (
        [Step(value) for value in args.select]
        if args.select is not None
        else [value for value in Step]
    )
    skips = [Step(value) for value in args.skip] if args.skip is not None else []

    repo_root = pathlib.Path(__file__).parent

    if Step.BLACK in selects and Step.BLACK not in skips:
        print("Black'ing...")
        # fmt: off
        black_targets = [
            "crosshair",
            "precommit.py",
            "setup.py",
            "check_init_and_setup_coincide.py"
        ]
        # fmt: on

        if overwrite:
            subprocess.check_call(["black"] + black_targets, cwd=str(repo_root))
        else:
            subprocess.check_call(
                ["black", "--check"] + black_targets, cwd=str(repo_root)
            )
    else:
        print("Skipped black'ing.")

    if Step.FLAKE8 in selects and Step.FLAKE8 not in skips:
        print("Flake8'ing...")
        # fmt: off
        subprocess.check_call(
            [
                "flake8", "crosshair", "--count",
                "--show-source",
                "--statistics"
            ],
            cwd=str(repo_root)
        )
        # fmt: on
    else:
        print("Skipped flake8'ing.")

    if Step.ISORT in selects and Step.ISORT not in skips:
        print("isorting'ing...")
        # fmt: off
        isort_cmd = ["isort", "crosshair"]
        if not overwrite:
            isort_cmd.extend("--diff")
        subprocess.check_call(
            isort_cmd,
            cwd=str(repo_root)
        )
        # fmt: on
    else:
        print("Skipped isort'ing.")

    if Step.PYDOCSTYLE in selects and Step.PYDOCSTYLE not in skips:
        print("Pydocstyle'ing...")
        subprocess.check_call(
            [
                "pydocstyle",
                "--ignore=D1,D203,D212",
                r"--match=.*(?<!_test).py$",
                r"--match-dir=(?!/examples/)",
                "crosshair",
            ],
            cwd=str(repo_root),
        )
    else:
        print("Skipped pydocstyle'ing.")

    if Step.MYPY in selects and Step.MYPY not in skips:
        print("Mypy'ing...")
        subprocess.check_call(
            [
                "mypy",
                ".",
            ],
            cwd=str(repo_root),
        )
    else:
        print("Skipped mypy.")

    if Step.DOCTEST in selects and Step.DOCTEST not in skips:
        # We doctest the documentation in a separate step from testing so that
        # the two steps can run in isolation.
        #
        # It is indeed possible to doctest the documentation *together* with
        # the other tests using pytest, but this is not desirable as tests can
        # take quite long to run. This would slow down the development if all we
        # want is to iterate on documentation doctests.
        print("Doctesting...")
        doc_source_dir = repo_root / "doc" / "source"
        for pth in (doc_source_dir).glob("**/*.rst"):
            subprocess.check_call([sys.executable, "-m", "doctest", str(pth)])
        subprocess.check_call([sys.executable, "-m", "doctest", "README.md"])
    else:
        print("Skipped doctesting.")

    if (
        Step.CHECK_INIT_AND_SETUP_COINCIDE in selects
        and Step.CHECK_INIT_AND_SETUP_COINCIDE not in skips
    ):
        print("Checking that crosshair/__init__.py and setup.py coincide...")
        subprocess.check_call([sys.executable, "check_init_and_setup_coincide.py"])
    else:
        print("Skipped checking that crosshair/__init__.py and " "setup.py coincide.")

    if Step.CHECK_HELP_IN_DOC in selects and Step.CHECK_HELP_IN_DOC not in skips:
        cmd = [sys.executable, "check_help_in_doc.py"]
        if overwrite:
            cmd.append("--overwrite")

        if not overwrite:
            print("Checking that --help's and the doc coincide...")
        else:
            print("Overwriting the --help's in the doc...")

        subprocess.check_call(cmd)
    else:
        print("Skipped checking that --help's and the doc coincide.")

    if Step.TEST in selects and Step.TEST not in skips:
        print("Testing...")
        env = os.environ.copy()
        # For determinism:
        env["PYTHONHASHSEED"] = "0"

        # fmt: off
        subprocess.check_call(
            [
                "python",
                "-m", "pytest",
                "--doctest-modules",
            ],
            cwd=str(repo_root),
            env=env,
        )
        # fmt: on
    else:
        print("Skipped testing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
