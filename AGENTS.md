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
- **`doc/source/`** - documentation (log user-facing changes in `changelog.rst`)
## Development Environment

The repo ships a **devcontainer** (`.devcontainer/`) that matches CI (dev deps, pre-commit hooks, pyenv-managed Python). It's the recommended way to run agents but not required — the notes below apply in any environment. See `.devcontainer/README.md` for host setup and what the container shares with your host.

- **Tests** run with `PYTHONHASHSEED=0` for reproducibility:
  - Smoke tests: `PYTHONHASHSEED=0 python -m pytest -m smoke -n auto --dist=worksteal`
  - Full suite: `PYTHONHASHSEED=0 python -m pytest -n auto --dist=worksteal crosshair`
- **CrossHair is very sensitive to Python version differences**. After changing the active interpreter, reinstall with `pip install -e .[dev]` so the `_crosshair_tracers` C extension is rebuilt for it — a stale `.so` from another version causes confusing failures. Inside the devcontainer, `switch-python <version>` does the install-and-reinstall in one step (see `.devcontainer/README.md`).

## Key Conventions

- **Formatting**: Black (88 chars), isort, flake8
- **Tests**: pytest; run with `PYTHONHASHSEED=0` for reproducibility
- **Pre-commit** runs black, isort, flake8, mypy, and pytest
- **Type annotations**: Required for all non-test code. Generally avoid type annotations in tests.
- **Naming and doc strings**: Name functions and parameters by what they **do**, not by how they're used. Rename as function behaviors evolve. Doc strings should not include historical context or litigate design decisions. Describe current behaviors only.
- **Code comments**: Use a **very high bar** - genuinely surpising or confusing behaviors only.

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
  - **Ensure tracing is on during symbolic operations.** ALL operations that involve symbolics require tracing. You'll likey get an exception if tracing isn't on. Examples that require tracing:
    - `len(symbolic_list)`
    - `symbolic_int > 0`
    - `next(symbolic_iter)`
  - CrossHair patching is **not like regular monkey-patching**
    - You never need import the "original" version of some function - the function never changes.
    - CrossHair uses function identity to intercept calls
  - To call the unpatched version of a function, you can either call it directly from the function body of its patch, or disable tracing.
  - Nest NoTracing and ResumedTracing blocks inside each other to toggle tracing. It's ok to nest NoTracing inside NoTracing (or ResumedTracing inside ResumedTracing), but the inner block effectively does nothing.
  - The symbolic-vs-conrete duality complicates type annotations. Our convention: annotate parameters according to the tracing state you expect. For example, a function that takes a (symbolic or concrete) int is:
    - Annotated with `Union[int, SymbolicInt]` when tracing will be off
    - Annotated with `int` when tracing will be on
  Use `# type: ignore` at tracing boundaries as needed.
- How to add symbolic support for something
  - **Consider leaving tracing on** – disabling gives a speedup but is error-prone. C-level code is often patched with plain Python + tracing.
  - In general, **avoid branching**. `x or y` implicitly creates a branch; instead, use something without short-circuiting like `x | y` or `any([x, y])`. Similarly, consider `(cond) * value` instead of `value if cond else 0`.
  - OTOH, consider **adding branching** to enable simpler execution paths or simpler SMT expressions. A central design tension in CrossHair is trading execution paths to keep solve times managable. (especially nonlinear integer arithmetic — floor division, multiplication of unknowns) Measure, don't assume.
  - Consider realizing symbolics early when we know they can't remain symbolic. (an integer that determines a future loop iteration)
  - Re-order operations to retain a larger state space for longer. Push work towards values that are likely to be concrete.
- Testing
  - **Unit tests start without tracing.**
    – If you use a `space` fixture parameter, you can `with ResumedTracing():` to enable tracing.
    - Otherwise, enter a statespace context to begin tracing.
  - To assert something is symbolic, query both sides for satisfiability: `assert space.is_possible(x == a)` and `assert space.is_possible(x != a)`.
  - Most pytest nicities (value printing!) are disabled because operating on symbolics under failure modes tends to obscure problems.
  - The test suite is large; use `pytest -n auto --dist=worksteal` when running several tests (`pytest-xdist` is already in dev dependencies).
  - Consider **debugging individual tests with `pytest -v`** - CrossHair's logging is verbose, but will display some important events:
    - When and where SMT checks occur, including the SMT expression.
    - When and where a value is realized.
    - Information at the end of each path exploration.
  - Use pytest.mark.parameterize when it's useful to test several cases
  - Avoid type annotations in test files except when important for testing. (tracing alternations are common and aren't worth the necessary typing workarounds)
