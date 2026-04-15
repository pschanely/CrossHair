# CrossHair – Agent Guide

CrossHair is a Python analysis tool that uses **symbolic execution** and an **SMT solver** (Z3) to explore execution paths. Use it to find examples that violate a code contract, generate tests, or find behavioral differences.

## Architecture Overview

- **`crosshair/main.py`** – CLI entry point, argument parsing, and orchestration
- **`crosshair/core.py`** – Core symbolic execution engine; runs functions with symbolic inputs and checks contracts
- **`crosshair/statespace.py`** – State space exploration, path splitting, and `MessageType` (CONFIRMED, POST_FAIL, etc.)
- **`crosshair/condition_parser.py`** – Parses pre/post contract conditions from docstrings and annotations
- **`crosshair/libimpl/`** – Symbolic implementations of stdlib modules (json, math, collections, etc.). Two test patterns:
  - **`xxxlib_test.py`** – Exercise the symbolic impl with direct tests: use `standalone_statespace` if possible, or `check_states` when path exploration is needed.
  - **`xxxlib_ch_test.py`** – CrossHair-on-CrossHair: define contract-checked functions (often using `compare_results`) and run `run_checkables(analyze_function(fn))` so CrossHair verifies the symbolic impl against the real stdlib
- **`crosshair/tracers.py`** – Bytecode tracing for symbolic execution (C extension in `_tracers.c`)
- **`crosshair/smtlib.py`**, **`crosshair/z3util.py`** – Z3/SMT integration

## Key Conventions

- **Formatting**: Black (88 chars), isort, flake8
- **Tests**: pytest; run with `PYTHONHASHSEED=0` for reproducibility
- **Pre-commit** runs black, isort, flake8, mypy, and pytest

## Working in This Codebase

- Symbolic values flow through `StateSpace`; use `realize()` / `deep_realize()` to get concrete values.
- Contract syntax is documented in `doc/source/kinds_of_contracts.rst`
- `libimpl` modules mirror stdlib APIs but operate on symbolic types
- Use `with NoTracing():` and `with ResumedTracing():` to toggle symbolic behavior:
  - **Tracing enables overrides** – function patches (`register_patch`, `register_type`) and bytecode overrides (`opcode_intercept.py`). Patches run with tracing on.
  - **`isinstance` and `type` depend on tracing** – symbolic values report their emulated type when tracing is on.
  - **Unit tests start with tracing disabled** – when using the `space` fixture, use `ResumedTracing` to enable.
  - **Ensure tracing is on when performing an operation on a symbolic.** E.g. `len(symbolic_list)`, `symbolic_int > 0`, `next(symbolic_iter)`; they may not function correctly without tracing.
  - **Consider leaving tracing on** – disabling gives a speedup but is error-prone. C-level code is often patched with plain Python with tracing enabled.
