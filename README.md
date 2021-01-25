<img src="doc/logo.png" width="5%" align="left">

# CrossHair

[![Join the chat at https://gitter.im/Cross_Hair/Lobby](https://badges.gitter.im/Cross_Hair/Lobby.svg)](https://gitter.im/Cross_Hair/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![codecov](https://codecov.io/gh/pschanely/CrossHair/branch/master/graph/badge.svg)](https://codecov.io/gh/pschanely/CrossHair)
[![Build Status](https://travis-ci.org/pschanely/CrossHair.svg?branch=master)](https://travis-ci.org/pschanely/CrossHair)

An analysis tool for Python that blurs the line between testing and type systems.

> **_THE LATEST NEWS:_**  CrossHair is growing new commands that are unrelated
to contracts; compare the behavior of two functions with the
[diffbehavior](doc/diff_behavior.md) command!

If you have a function with
[type annotations](https://www.python.org/dev/peps/pep-0484/) and add a
contract [in a supported syntax](#kinds-of-contracts),
CrossHair will attempt to find counterexamples for you:

![Animated GIF demonstrating the verification of a python function](doc/duplicate_list.gif)

CrossHair works by repeatedly calling your functions with symbolic inputs.
It uses an [SMT solver](https://en.wikipedia.org/wiki/Satisfiability_modulo_theories) (a kind of theorem prover) to explore viable execution paths and find counterexamples for you.
This is not a new idea; an approach for Python was first described in [this paper](https://hoheinzollern.files.wordpress.com/2008/04/seer1.pdf).
However, to my knowledge, CrossHair is the most complete implementation of the idea: it supports symbolic lists, dictionaries, sets, and custom/mutable objects.

Try CrossHair right now, in your browser, at [crosshair-web.org](https://crosshair-web.org)!

> **_NOTE:_**  CrossHair is in an experimental state right now. You can help though - keep reading!

|Contents|
|--------|
|[Kinds of Contracts](#kinds-of-contracts)|
|[Why Should I Use CrossHair?](#why-should-i-use-crosshair)|
|[Get Started](#get-started)|
|[IDE Integrations](#ide-integrations)|
|[Limitations](#limitations)|
|[How Can I Help?](#how-can-i-help)|
|[Related Work](#related-work)|
|[Contributors](#contributors)|
|[Change Log](#change-log)|

## Kinds of Contracts

|`analysis_kind=`||
|---------:|:-------|
|`asserts`|The lowest-friction way to get started with CrossHair. No imports, no new syntax; just use regular Python assert statements. [(more details)](doc/analysis_kind_asserts.md)|
|`PEP316`|Docstring-based. Compact and doesn't require a library, but there's some syntax to learn. [(more details)](doc/analysis_kind_pep316.md)|
|`icontract`|Decorator-based. Contracts are in regular Python and can leverage your IDE's autocomplete. [(more details)](doc/analysis_kind_icontract.md)|


## Why Should I Use CrossHair?

These examples use the PEP316 format, but the motivation applies to all
contract kinds.

**More precision.** Commonly, we care about more than just the type. Is it really any integer, or is it a **positive** integer? Is it any list, or does it have to be a non-empty list? CrossHair gives you that precision:

![Image showing an average function](doc/average.png)

**Inter-procedural analysis.** CrossHair (1) validates the pre-conditions of called functions and (2) uses post-conditions of called functions to help it prove post-conditions in the caller.

![Image showing CrossHair caller and callee](doc/zipped_pairs.png)

**Verify across all implementations.** Contracts are particularly helpful when applied to base classes / interfaces: all implementations will be verified against them:

![Image showing CrossHair contract and inheritance](doc/chess_pieces.png)

**Catch errors.** Setting a trivial post-condition of "True" is enough to enable analysis, which will find exceptions like index out-of-bounds errors:

![Image showing CrossHair contract and IndexError](doc/index_bounds.gif)

**Support your type checker.** CrossHair is a nice companion to [mypy](http://mypy-lang.org/). Assert statements divide work between the two systems:

![Image showing mypy and CrossHair together](doc/pair_with_mypy.png)

**Optimize with Confidence.** Using post-conditions, CrossHair ensures that optimized code continues to behave like equivalent naive code:

![Image showing the equivalence of optimized an unoptimized code](doc/csv_first_column.png)

## Get Started

> **_NOTE:_**  CrossHair is in an experimental state right now. If you're using it, it's because you want it to succeed, want to help, or are just interested in the technology.

CrossHair is supported only on Python 3.7+ and only on CPython (the most common Python implementation).

Inside the development environment of the code you want to analyze (virtual environment, conda environment, etc), install:
```shell
pip install crosshair-tool
```

CrossHair works best when it sits in its own window and thinks about your code while you work on it. Open such a window, activate your development environment, and run:
```shell
crosshair watch [directory with code to analyze]
```
You should then see periodically updating text as CrossHair analyzes the contracts in your code. It will watch for changes and re-analyze as appropriate. When it detects an issue, you'll see something like this:

![Image showing terminal output](doc/example_error.png)

Hit Ctrl-C to exit.

## IDE Integrations

As mentioned above, CrossHair wants to run in the background so it can have plenty of time to think. However, IDE integrations can still be used to catch easy cases.

* [Emacs (flycheck)](https://github.com/pschanely/emacs-flycheck-crosshair)

If you make a plugin for your favorite editor (please do!), submit a pull request to add it above. The `crosshair check [FILENAME]` command will yield results in the same format as the mypy type checker. (a non-zero exit for errors, and lines formatted as `{FILENAME}:{LINE_NUMBER}:error:{MESSAGE}`)

## Limitations

A (wildly incomplete) list of present limitations. Some of these will be lifted over time (your help is welcome!); some may never be lifted.

* Be aware that the absence of a counterexample does not guarantee that the property holds.
* Symbolic values are implemented as Python proxy values. CrossHair monkey-patches the system to maintain a good illusion, but the illusion is not complete. For example,
  * Code that cares about the identity values (x is y) may not be correctly analyzed.
  * Code that cares about the types of values may not be correctly analyzed.
* Only function and class definitions at the top level are analyzed. (i.e. not when nested inside other functions/classes)
* Only deterministic behavior can be analyzed. (your code always does the same thing when starting with the same values)
  * CrossHair may produce a `NotDeterministic` error when it detects this.
* Be careful: CrossHair will actually run your code and may apply any arguments to it.
  * If you run CrossHair on code calling [shutil.rmtree](https://docs.python.org/3/library/shutil.html#shutil.rmtree), you **will** destroy your filesystem.
* Consuming values of an iterator/generator in a pre- or post-condition will produce [unexpected behavior](https://github.com/pschanely/CrossHair/issues/9).
* SMT solvers have very different perspectives on hard problems and easy problems than humans.
  * Be prepared to be surprised both by what CrossHair can tell you, and what it cannot.

## How Can I Help?

* [Try it out](#get-started) on your own python project! Be communicative about what does and doesn't work.
* Participate (or just lurk) in the [gitter chat](https://gitter.im/Cross_Hair/Lobby).
* [File an issue](https://github.com/pschanely/CrossHair/issues).
* [Ask a question](https://stackoverflow.com/questions/tagged/crosshair) on stackoverflow.
* Make a pull request. There aren't contributing guidelines yet - just check in on [gitter](https://gitter.im/Cross_Hair/Lobby) to coordinate.
* Help me evangelize: Share with your friends and coworkers. If you think it's neato, star the repo. :star:
* Contact me at `pschanely@gmail.com` or [Twitter](https://twitter.com/pschanely)... even if it's just to say that you'd like me to cc you on future CrossHair-related developments.

## Related Work

|Technology|Relation|
|---------:|:-------|
| [dependent types](https://en.wikipedia.org/wiki/Dependent_type), [refinement types](https://en.wikipedia.org/wiki/Refinement_type) | CrossHair aims to provide many of the same capabilities as these advanced type systems. CrossHair is easier to learn (because it is just python), but is incomplete (it can't always tell you whether a condition holds). |
| [design by contract](https://en.wikipedia.org/wiki/Design_by_contract) | Unlike most tools for contracts, CrossHair attempts to verify pre-conditions and post-conditions before you run them. |
| [fuzz testing](https://en.wikipedia.org/wiki/Fuzzing), [QuickCheck](https://en.wikipedia.org/wiki/QuickCheck), [property testing](https://en.wikipedia.org/wiki/Property_testing), [Hypothesis](https://hypothesis.readthedocs.io/) | CrossHair has many of the same goals as these tools. However, CrossHair uses an SMT solver to find inputs rather than the (typically) randomized approaches that these tools use. |
| [concolic testing](https://en.wikipedia.org/wiki/Concolic_testing) | State-of-the-art fuzz testers employ SMT solvers in a similar fashion as CrossHair. |
| [SMT solvers](https://en.wikipedia.org/wiki/Satisfiability_modulo_theories) | SMT solvers power many of the tools in this table. CrossHair uses [Z3](https://github.com/Z3Prover/z3). |
| [angr](https://angr.io), [klee](https://klee.github.io/) | Symbolic execution of **binary** code. Unlike these tools, CrossHair models the semantics of Python directly. |
| [PyExZ3](https://github.com/thomasjball/PyExZ3), [pySim](https://github.com/bannsec/pySym), [PEF](https://git.cs.famaf.unc.edu.ar/dbarsotti/pef) | Take approaches that are very similar to CrossHair, in various states of completeness. CrossHair is generally more prescriptive or product-like than these tools. |

## Contributors

* [**Phil Schanely**](https://twitter.com/pschanely)
* [**Edward Haigh**](https://github.com/oneEdoubleD)
* [**Saul Shanabrook**](https://github.com/saulshanabrook/)

## Change Log

0.0.9
* Introduce [the diffbehavior command](doc/diff_behavior.md) which finds inputs that distinguish the behavior of two functions.
* Upgrade to the latest release of z3 (4.8.9.0)
* Fix [an installation error](https://github.com/pschanely/CrossHair/issues/41) on Windows.
* Fix a variety of other bugs.
