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
- **`doc/source/`** - documentation (log user-facing changes in `changelog.rst`). CLI `--help` text lives in `crosshair/main.py`; the `check_help_in_doc` pre-commit hook regenerates the embedded help blocks in `contracts.rst`, `cover.rst`, and `diff_behavior.rst`. Edit the help strings in `main.py` and run pre-commit — don't hand-edit those `.rst` snippets.
## Development Environment

The repo ships a **devcontainer** (`.devcontainer/`) that matches CI (dev deps, pre-commit hooks, pyenv-managed Python). It's the recommended way to run agents but not required — the notes below apply in any environment. See `.devcontainer/README.md` for host setup and what the container shares with your host.

- **Tests** run with `PYTHONHASHSEED=0` for reproducibility:
  - Smoke tests: `PYTHONHASHSEED=0 python -m pytest -m smoke -n auto --dist=worksteal`
  - Full suite: `PYTHONHASHSEED=0 python -m pytest -n auto --dist=worksteal crosshair`
  - The full parallel suite can report **spurious** failures in symbolic/time-sensitive tests under local load. Re-run a failing nodeid in isolation (with `PYTHONHASHSEED=0`) before treating it as real — CI is the source of truth for the full matrix. `main` itself is occasionally red for unrelated path-exploration/datetime reasons, so check whether a failure also reproduces on unmodified `main`.
  - Tests that mutate process-global state (the type repository, type discovery) must restore it within the test — xdist reuses workers, so leaked state poisons later tests in the same worker.
- **CrossHair is very sensitive to Python version differences**. After changing the active interpreter, reinstall with `pip install -e .[dev]` so the `_crosshair_tracers` C extension is rebuilt for it — a stale `.so` from another version causes confusing failures. Inside the devcontainer, `switch-python <version>` does the install-and-reinstall in one step (see `.devcontainer/README.md`).

## Key Conventions

- **Formatting**: Black (88 chars), isort, flake8
- **Tests**: pytest; run with `PYTHONHASHSEED=0` for reproducibility
- **Pre-commit** runs black, isort, flake8, mypy, and pytest
- **Code comments**: Use a high bar - genuinely surpising or confusing behaviors only.

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

## Performance & the C tracer

The tracer hot path is being incrementally ported from Python (`tracers.py`) to the `_crosshair_tracers` C extension (`_tracers.c` / `_tracers.h`; tracking issue #115). When touching this area:

- **Don't add logic to the Python side of the hot path if it belongs in C.** Python-side micro-optimizations to code slated to move into the C tracer get rejected even when they're correct and measurably faster — the maintainer would rather not carry Python logic that's about to be deleted.
- **Never add a "primitive/native type" fast-path that skips interception.** CrossHair must intercept method calls on native instances (e.g. to swap in symbolic-tolerant implementations), so short-circuiting tracing/normalization for builtin types is a correctness bug, not an optimization.
- In C, after a failed attribute/descriptor lookup, clear the `AttributeError` but **propagate any other exception** — never continue with an exception still set (risks crashes/corruption).
- After editing C, rebuild with `pip install -e .[dev]`, then validate with `crosshair/tracers_test.py` + `crosshair/_tracers_test.py`, followed by `crosshair/core_test.py` + `crosshair/libimpl/builtinslib_test.py` (optionally `CROSSHAIR_EXTRA_ASSERTS=1`).
- **Benchmark each optimization in isolation** and report before/after numbers — bundled "optimize everything" changes aren't reviewable and often don't hold up. Alternative representations sometimes measure *slower*; when you reject one, record the measurement in a code comment near the relevant function (see the rejected ordinal-bridge note in symbolic `datetime` support).

## Contributing Workflow

See `doc/source/contributing.rst` for the full guide. Points that repeatedly matter for agents:

- **Coordinate first.** Several contributors race on the tracer/perf work, and overlapping changes get discarded. Before coding, comment on the relevant issue to claim a narrow, specific slice and confirm scope; reference sibling PRs to show you don't overlap them.
- **Keep PRs small and single-purpose, and state explicitly what you are _not_ changing.** Broad PRs get picked apart and stall.
- **Respond to review promptly.** PRs are closed for post-review inactivity, and unaddressed feedback gets the same slice handed to another contributor. Passing tests don't excuse leaving a maintainer's question unanswered.
- Run formatting/linting (black, isort, flake8, mypy) before submitting, and add yourself to the contributor list at the bottom of `contributing.rst`.
