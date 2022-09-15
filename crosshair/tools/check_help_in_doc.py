#!/usr/bin/env python3

"""Check that the help snippets in the doc coincide with the actual output."""
import argparse
import os
import pathlib
import re
import subprocess
import sys
from typing import List, Optional, Tuple

import icontract


class Block:
    """Represent a block in the readme that needs to be checked."""

    @icontract.require(lambda command: command != "")
    @icontract.require(
        lambda start_line_idx, end_line_idx: start_line_idx <= end_line_idx
    )
    def __init__(self, command: str, start_line_idx: int, end_line_idx: int) -> None:
        """
        Initialize with the given values.

        :param command: help command
        :param start_line_idx: index of the first relevant line
        :param end_line_idx: index of the first line excluded from the block
        """
        self.command = command
        self.start_line_idx = start_line_idx
        self.end_line_idx = end_line_idx


HELP_STARTS_RE = re.compile(r"^.. Help starts: (?P<command>.*)$")


def parse_rst(lines: List[str]) -> Tuple[List[Block], List[str]]:
    """
    Parse the code blocks that represent help commands in the RST file.

    :param lines: lines of the readme file
    :return: (help blocks, errors if any)
    """
    blocks = []  # type: List[Block]
    errors = []  # type: List[str]

    i = 0
    while i < len(lines):
        mtch = HELP_STARTS_RE.match(lines[i])
        if mtch:
            command = mtch.group("command")
            help_ends = ".. Help ends: {}".format(command)
            try:
                end_index = lines.index(help_ends, i)
            except ValueError:
                end_index = -1

            if end_index == -1:
                return [], ["Could not find the end marker {!r}".format(help_ends)]

            blocks.append(
                Block(command=command, start_line_idx=i + 1, end_line_idx=end_index)
            )

            i = end_index + 1

        else:
            i += 1

    return blocks, errors


def capture_output_lines(command: str) -> List[str]:
    """Capture the output of a help command."""
    command_parts = command.split(" ")
    if command_parts[0] in ["python", "python3"]:
        # We need to replace "python" with "sys.executable" on Windows as the environment
        # is not properly inherited.
        command_parts[0] = sys.executable

    proc = subprocess.Popen(
        command_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    output, err = proc.communicate()
    if err:
        raise RuntimeError(
            f"The command {command!r} failed with exit code {proc.returncode} and "
            f"stderr:\n{err}"
        )
    # Help text changed in 3.10 argparse; always use the newer text.
    output = output.replace("optional arguments", "options")

    return output.splitlines()


def output_lines_to_code_block(output_lines: List[str]) -> List[str]:
    """Translate the output of a help command to a RST code block."""
    result = (
        [".. code-block:: text", ""]
        + ["    " + output_line for output_line in output_lines]
        + [""]
    )

    result = [line.rstrip() for line in result]
    return result


def diff(got_lines: List[str], expected_lines: List[str]) -> Optional[str]:
    """
    Report a difference between the ``got`` and ``expected``.

    Return None if no difference.
    """
    if got_lines == expected_lines:
        return None

    result = []

    result.append("Expected:")
    for i, line in enumerate(expected_lines):
        if i >= len(got_lines) or line != got_lines[i]:
            print("DIFF: {:2d}: {!r}".format(i, line))
        else:
            print("OK  : {:2d}: {!r}".format(i, line))

    result.append("Got:")
    for i, line in enumerate(got_lines):
        if i >= len(expected_lines) or line != expected_lines[i]:
            print("DIFF: {:2d}: {!r}".format(i, line))
        else:
            print("OK  : {:2d}: {!r}".format(i, line))

    return "\n".join(result)


def process_file(path: pathlib.Path, overwrite: bool) -> List[str]:
    """
    Check or overwrite the help blocks in the given file.

    :param path: to the doc file
    :param overwrite: if set, overwrite the help blocks
    :return: list of errors, if any
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    blocks, errors = parse_rst(lines=lines)
    if errors:
        return errors

    if len(blocks) == 0:
        return []

    if overwrite:
        result = []  # type: List[str]

        previous_block = None  # type: Optional[Block]
        for block in blocks:
            output_lines = capture_output_lines(command=block.command)
            code_block_lines = output_lines_to_code_block(output_lines=output_lines)

            if previous_block is None:
                result.extend(lines[: block.start_line_idx])
            else:
                result.extend(lines[previous_block.end_line_idx : block.start_line_idx])

            result.extend(code_block_lines)
            previous_block = block
        assert previous_block is not None
        result.extend(lines[previous_block.end_line_idx :])
        result.append("")  # new line at the end of file

        path.write_text("\n".join(result))
    else:
        for block in blocks:
            output_lines = capture_output_lines(command=block.command)
            code_block_lines = output_lines_to_code_block(output_lines=output_lines)

            expected_lines = lines[block.start_line_idx : block.end_line_idx]
            expected_lines = [line.rstrip() for line in expected_lines]

            error = diff(got_lines=code_block_lines, expected_lines=expected_lines)
            if error:
                return [error]
    return []


def main() -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        help="If set, overwrite the relevant part of the doc in-place.",
        action="store_true",
    )

    args = parser.parse_args()
    overwrite = bool(args.overwrite)

    this_dir = pathlib.Path(os.path.realpath(__file__)).parent.parent.parent

    pths = [
        this_dir / "doc" / "source" / "contracts.rst",
        this_dir / "doc" / "source" / "cover.rst",
        this_dir / "doc" / "source" / "diff_behavior.rst",
        this_dir / "doc" / "source" / "contributing.rst",
    ]

    success = True

    for pth in pths:
        errors = process_file(path=pth, overwrite=overwrite)
        if errors:
            print("One or more errors in {}:".format(pth), file=sys.stderr)
            for error in errors:
                print(error, file=sys.stderr)
            success = False

    # Also check that the TOC in the README matches the Sphinx TOC:
    indexlines = open(this_dir / "doc" / "source" / "index.rst").readlines()
    rst_links = [
        f"latest/{line.strip()}.html"
        for line in indexlines
        if re.fullmatch(r"\s*[a-z_]+\s*", line)
    ]
    readme_text = open(this_dir / "README.md").read()
    readme_idx = readme_text.index(
        "## [Documentation]"
    )  # find the Documentation section
    readme_links = list(re.findall(r"latest/\w+.html", readme_text[readme_idx:]))
    if rst_links != readme_links:
        success = False
        chapters_only_in_rst = set(rst_links) - set(readme_links)
        chapters_only_in_readme = set(readme_links) - set(rst_links)
        if chapters_only_in_rst:
            print(
                f"Error: chapters in index.rst, but missing from README.md: {list(chapters_only_in_rst)}",
                file=sys.stderr,
            )
        elif chapters_only_in_readme:
            print(
                f"Error: chapters in README.md, but missing from index.rst: {list(chapters_only_in_readme)}",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: chapters in README.md and index.rst have different orderings. {rst_links} != {readme_links}",
                file=sys.stderr,
            )

    if not success:
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())
