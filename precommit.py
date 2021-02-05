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
    PYDOCSTYLE = "pydocstyle"
    TEST = "test"
    DOCTEST = "doctest"
    CHECK_INIT_AND_SETUP_COINCIDE = "check-init-and-setup-coincide"


def main() -> int:
    """"Execute entry_point routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        help="Overwrites the unformatted source files with the well-formatted code in place. "
        "If not set, an exception is raised "
        "if any of the files do not conform to the style guide.",
        action="store_true",
    )
    parser.add_argument(
        "--select",
        help="If set, only the selected steps are executed. "
        "This is practical if some of the steps failed and you want to fix them in isolation.",
        nargs="+",
        choices=[value.value for value in Step],
    )
    parser.add_argument(
        "--skip",
        help="If set, skips the specified steps. "
        "This is practical if some of the steps passed and "
        "you want to fix the remainder in isolation.",
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
                "flake8", "crosshair", "--count", "--select=E9,F63,F7,F82",
                "--show-source",
                "--statistics"
            ],
            cwd=str(repo_root)
        )
        # fmt: on
    else:
        print("Skipped flake8'ing.")

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

    if Step.TEST in selects and Step.TEST not in skips:
        print("Testing...")
        env = os.environ.copy()
        env["ICONTRACT_SLOW"] = "true"

        # fmt: off
        subprocess.check_call(
            [
                "coverage", "run",
                "--source", "crosshair",
                "--omit=__init__.py"
                "--omit=*_test.py",
                "-m", "pytest",
                "--doctest-modules"
            ],
            cwd=str(repo_root),
            env=env,
        )
        # fmt: on

        subprocess.check_call(["coverage", "report"])
    else:
        print("Skipped testing.")

    if Step.DOCTEST in selects and Step.DOCTEST not in skips:
        # We doctest the documentation in a separate step from testing so that
        # the two steps can run in isolation.
        #
        # It is indeed possible to doctest the documentation *together* with
        # the other tests using pytest (and even measure the code coverage),
        # but this is not desirable as tests can take quite long to run.
        # This would slow down the development if all we want is to iterate
        # on documentation doctests.
        print("Doctesting...")
        for pth in (repo_root / "doc").glob("**/*.md"):
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
