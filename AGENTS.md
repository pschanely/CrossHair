# CrossHair – Agent Guide

CrossHair is a Python analysis tool that uses **symbolic execution** and an **SMT solver** (Z3) to explore execution paths. Use it to find examples that violate a code contract, generate tests, or find behavioral differences.

## Architecture Overview

- **`crosshair/main.py`** – CLI entry point, argument parsing, and orchestration
- **`crosshair/core.py`** – Core symbolic execution engine; runs functions with symbolic inputs and checks contracts
- **`crosshair/statespace.py`** – State space exploration, path splitting, and `MessageType` (CONFIRMED, POST_FAIL, etc.)
- **`crosshair/condition_parser.py`** – Parses pre/post contract conditions from docstrings and annotations
- **`crosshair/libimpl/`** – Symbolic implementations of stdlib modules (json, math, collections, etc.). Two test patterns:
  - **`xxxlib_test.py`** – Exercise the symbolic impl with direct tests: use the `state` fixture if possible, or `check_states` when path exploration is needed.
  - **`xxxlib_ch_test.py`** – CrossHair-on-CrossHair: define contract-checked functions (often using `compare_results`) and run `run_checkables(analyze_function(fn))` so CrossHair verifies the symbolic impl against the real stdlib
- **`crosshair/tracers.py`** – Bytecode tracing for symbolic execution (C extension in `_tracers.c`)
- **`crosshair/smtlib.py`**, **`crosshair/z3util.py`** – Z3/SMT integration

## Key Conventions

- **Formatting**: Black (88 chars), isort, flake8
- **Tests**: pytest; run with `PYTHONHASHSEED=0` for reproducibility
- **Pre-commit** runs black, isort, flake8, mypy, and pytest

## Must-Know Technical Background

- CrossHair works by repeatedly calling the target function to explore execution paths.
- It inspects and mutates Python behaviors via a standard sys.monitoring tracer; it has a system for patching functions with symbolic-friendly versions, and can mutate the interpreter stack when detecting certain bytecodes.
- All solver operations go through a `StateSpace` context variable, which also tracks the current execution path.
- Use `realize()` / `deep_realize()` to get concrete values out of a symbolic.
- Contract-checking syntax is documented in `doc/source/kinds_of_contracts.rst`
- Use `with NoTracing():` and `with ResumedTracing():` to toggle symbolic behavior inside a block:
  - **`isinstance` and `type` depend on tracing** – symbolic values report their emulated type when tracing is on.
  - **Tracing enables overrides/patches** – function patches (`register_patch`, `register_type`) and bytecode overrides (`opcode_intercept.py`). The code for a patch always starts with tracing on.
  - Add `@assert_tracing(True)` (or False) to clarify that your function expects a specific tracing status. (assert_tracing becomes enforced with a CROSSHAIR_EXTRA_ASSERTS=1 env var)
  - **Ensure tracing is on during symbolic operations.** ALL operations that involve symbolics require tracing. You will likey get an exception will usually fail if tracing isn't on. Examples that require tracing:
    - `len(symbolic_list)`
    - `symbolic_int > 0`
    - `next(symbolic_iter)`
  - CrossHair patching is **not like regular monkey-patching** - it intercepts calling bytecodes triggers on the identity of the invoked function.
  - To call the unpatched version of a function, you can either call it directly from the function body of its patch, or disable tracing.
  - **Consider leaving tracing on** – disabling gives a speedup but is error-prone. C-level code is often patched with plain Python with tracing enabled.
  - Nest NoTracing and ResumedTracing blocks inside each other to toggle tracing. It's ok to nest NoTracing inside NoTracing (or ResumedTracing inside ResumedTracing), but the inner block effectively does nothing.
- Testing
  - **Unit tests start without tracing.**
    – If you use a `space` fixture parameter, you can `with ResumedTracing():` to enable tracing.
    - Otherwise, enter a statespace context to begin tracing.
  - Most pytest nicities (value printing!) are disabled because operating on symbolics under failure modes tends to obscure problems.
  - Consider **debugging individual tests with `pytest -v`** - CrossHair's logging is verbose, but will display some important events:
    - When and where SMT checks occur, including the SMT expression.
    - When and where a value is realized.
    - Information at the end of each path exploration.
