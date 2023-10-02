#########
Changelog
#########


Next Version
------------

* Nothing yet!


Version 0.0.41
--------------

* Add ``--output_all_examples`` option for outputting every example with
  a new best score when optimizing with the ``crosshair search`` command.


Version 0.0.40
--------------

* Ensure that Ctrl-C is never considered an exception produced by the code under
  analysis.
  (fixes `#206 <https://github.com/pschanely/CrossHair/issues/206>`__)
* Make ``crosshair watch`` Show tracebacks for errors during import.
  (fixes `#202 <https://github.com/pschanely/CrossHair/issues/202>`__)
* Add ``--argument_formatter`` option to cusomtize the output of the
  ``crosshair search`` command.


Version 0.0.39
--------------

* Introduce path search heuristic based on code coverage.
* Optimize containment checks in symbolic strings.
  (fixes `#207 <https://github.com/pschanely/CrossHair/issues/207>`__)


Version 0.0.38
--------------

* Add a new (highly experimental) ``crosshair search`` command.
  Some people have been using ``crosshair check`` to look for counterexamples that
  they intend or expect to find. The ``crosshair search`` command is an easier and
  more featureful way to do this, and includes an option to search for inputs that
  score best along some objective.
  There is nothing in the official docs yet for this, but fiddle with it on the command
  line and give some feedback in a github discussion!
* The ``--example_output_format=argument_dictionary`` option for the cover command
  never actually output a dictionary! This is now fixed with the similarly named option
  ``--example_output_format=arg_dictionary``; the old option will issue a warning for a
  few releases and then be removed.


Version 0.0.37
--------------

* Avoid false positive counterexample when user code handles ``Exception``.
  (fixes `#196 <https://github.com/pschanely/CrossHair/issues/196>`__)
* Reduce path explosion when parsing integers from a string.
* Fix CrossHair build from source distribution (notably affects linux arm64 & Apple
  silicon).
  We still don't ship binary packages to PyPI (github actions still does not have
  runners?) but at least you should be able to build on your own.
  (Fixes `#197 <https://github.com/pschanely/CrossHair/issues/197>`__)


Version 0.0.36
--------------

* Add pygls 1.0 compatibility. (this is for the LSP server)


Version 0.0.35
--------------

* Complete Python 3.11 support!
* Add symbolic branch collapsing for ``any()`` and ``all()``. This can significantly
  reduce the number of branches to explore when these functions are applied to symbolic
  inputs.
* Preserve symbolic bools through the ``not`` operator.
* Fix premature path exhaustion when CrossHair attempts to generate ``TypedDict``
  instances inside containers.
  (see `this discussion <https://github.com/pschanely/CrossHair/discussions/193>`__)
* Fix crash when attempting to create an instance of a user-defined class that has an
  argument named ``typ``.
  (fixes `#191 <https://github.com/pschanely/CrossHair/issues/191>`__)


Version 0.0.34
--------------

* Save hypothesis counterexamples to the hypothesis database.
  Now, regular runs of hypothesis will try inputs that CrossHair has found.
  (thanks `Zac-HD <https://github.com/Zac-HD>`__!)
* Fix a regression in ``crosshair watch`` that crashes when the code under test attempts
  to print to stdout.
* Fix issue with the new C tracer that could result in the tracer unexpectedly remaining
  engaged.
* Require ``crosshair watch`` file arguments exist on disk at launch time.
  (they can still disappear/reappear during execution without issue, however)


Version 0.0.33
--------------

* Implement several optimizations; CrossHair is >2X faster on nearly all of the
  `official benchmarks <https://github.com/pschanely/crosshair-benchmark>`__!
* Switch to an opcode tracer written in C. Build binary wheels on major platforms.
* Optimize nondeterminism checking and z3 API usage; reuse SMT decisions.
* Fix regex bug: count chars #28-#31 as whitespace in unicode mode.
* Switch to use pre-commit for code checks. (no user-facing changes)
* Supply encoding for setup.py's open().
  (fixes `#179 <https://github.com/pschanely/CrossHair/issues/179>`__)


Version 0.0.32
--------------

* [**breaking change**] Change how custom classes are shown in counterexamples.
  Previously, CrossHair would call repr() on the instance of the custom class.
  Now, CrossHair will create an eval()able string that minics how CrossHair created the
  instance originally (and repr() is not used in the counterexample generation).
  (fixes `#164 <https://github.com/pschanely/CrossHair/issues/164>`__)
* [**breaking change**] Implement a different strategy for symbolic Callables.
  Now, symbolic callables simply invent a list of return values that are simply
  handed out, one at a time.
  This means that Callable counterexamples may be quite a bit more ugly.
  On the other hand, this new strategy fixes soundness issues and adds support for
  complex argument and return types. (only atomic types were supported previously)
* [**breaking change**] Make it easier to work with timeouts. Now, if you specify a
  ``--per_condition_timeout=`` parameter, CrossHair scales the ``--per_path_timeout=``
  default accordingly (namely, to the square root of the per_condition_timeout).
  That means just increasing the per_condition_timeout is sufficient uniformly scale up
  the amount of effort to put into a problem.
* (Finally!) Upgrade our z3 version to the latest (4.11.2.0).
  Reach out if you notice significant changes in your environments!
* Make some performance enhancements when type annotations are missing or incomplete
  (e.g. ``x: list`` instead of ``x: List[int]``).
* Add missing f-string support for formatting, e.g. ``f"{item!r}: {price:02d}"``.
* Fix issues in ``diffbehavior`` and ``cover`` where an ``IgnoreAttempt`` exception
  could escape and cause the process to abnormally exit.
* Fix a bug where ``splitlines()`` was not splitting on "\\r" characters.
* Fix a bug where CrossHair mistakenly evaluated ``" ".isprintable()`` to False.


Version 0.0.31
--------------

* LSP server: ensure the watcher thread has enough time to kill workers on shutdown.
* Fix bug in which str/repr for bytes objects returned the NotImplemented object.


Version 0.0.30
--------------

* Fix important issues with list concatenation and slicing: ensure arguments are
  always evaluated properly, and that results are real symbolic lists.
* Explicitly shut down the LSP server's worker pool when getting a shutdown message
  from the client. Reduces the possibility of leaked workers. Ensure your VSCode
  extension is updated, too!
* Unify comment parsing behavior for "raises" phrases in docstrings.
  (for consistency with other contract syntaxes, unparsable PEP316 raises phrases no
  longer produce syntax errors)
* Preserve symbolics across int-to-str conversions.
* Fix deque issues with extend(), extendleft(), and equality comparisons.
* Improve performance in counterexample generation and regex against literals.


Version 0.0.29
--------------

* Add support for symbolic containment checks in concrete dictionaries.
* Fix several issues with the LSP server on windows.
* Fix `cover` command errors when applied to wrapped functions and methods of
  dataclasses.


Version 0.0.28
--------------

* Do not manually set ``typing.TYPE_CHECKING`` to True.
  This is a **breaking change** - unfortunately, too many regular and correct typing
  guards will not work at runtime with TYPE_CHECKING on.
  (for one, you can use a guard to protect an import of a ``.pyi`` module,
  e.g. pytorch in `#172 <https://github.com/pschanely/CrossHair/issues/172>`__ )
  CrossHair will now only be able to understand types that are present and resolvable
  at runtime.
  (previously it might have been able to resolve types in more cases, e.g. the circular
  dependencies in `#32 <https://github.com/pschanely/CrossHair/issues/32>`__ )


Version 0.0.27
--------------

* Automatically disable ``lru_cache`` and ``cache`` decorations during analysis.
  (this prevents nondeterministic errors when analyzing code that uses them!)
* Disable side-effect detection when importing modules.
  (fixes `#172 <https://github.com/pschanely/CrossHair/issues/172>`__)
* Reduce path explosions when checking for symbolic string containment in a concrete
  string.
* Fix unexpected nondeterminism exception when calling ``urllib.parse``.
* Finish making sure ``unicodedata`` functions are tolerant to symbolic string arguments.
* Make ``heapq`` functions tolerant to symbolic list arguments.


Version 0.0.26
--------------

* Fix crash when running ``crosshair cover`` over functions that raise exceptions.
  (fixes `#171 <https://github.com/pschanely/CrossHair/issues/171>`__)
* Add symbolic handling when the callback used in ``map``, ``filter``,
  ``reduce``, or ``partial`` is native and intolerant to symbolics.
  (string functions, most commonly)
* Allow writes to the "nul" file on Windows.
* Add various preparations for Python 3.11.

Version 0.0.25
--------------

* Add the ``crosshair server`` command. This starts a Language Server Protocol (LSP)
  server that can simplify integration with several IDEs.
  (look for new versions of the VSCode extension that use this soon; consider
  contributing one for your favorite editor!)
* Present counterexamples that describe argument aliasing using the
  "walrus" operator, e.g. ``foo([a:=[], [], a])`` to describe a counterexample that
  takes a list of three empty sublists, where the frist and third are the same list.
  (fixes `#48 <https://github.com/pschanely/CrossHair/issues/48>`__)
  Note that CrossHair does not yet reliably detect all kinds of aliasing problems;
  see `this issue <https://github.com/pschanely/CrossHair/issues/47>`__ in particular.
* Fix code parse error over docstrings with blank lines.
* Fix bug when ``get()`` is called with a numeric symbolic key on a concrete
  dictionary.
* Fix crash when ``re.match()`` or ``re.finditer()`` is invoked on a sliced string.
* Ensure the ``key=`` function of ``itertools.groupby`` can be itercepted with
  ``register_patch()``.
* Correctly lowercase mid-word, mixed-case characters when titlecasing a string.
* Fix a crash when the checked code imports additional modules at runtime which define
  new namedtuples.


Version 0.0.24
--------------

* CrossHair can now invent symbolic return values for many calls like ``time.time`` and
  ``random.randrange``. See
  `this issue <https://github.com/pschanely/CrossHair/issues/162>`__ for what's
  supported.
* Allow subprocess spawning by standard library modules like ``uuid``, ``plaftorm``, and
  ``ctypes``. Previously, CrossHair would crash on some calls/platforms, complaining about
  side effects.
  (fixes `#163 <https://github.com/pschanely/CrossHair/issues/163>`__)


Version 0.0.23
--------------

* Add support for attaching a contract to an external function.
  Among other things, this can help you check code involving nondeterministic functions
  like ``time.time()``.
  See `the docs <https://crosshair.readthedocs.io/en/latest/plugins.html#adding-contracts-to-external-functions>`__
  for all the details.
  (thanks to `lmontand <https://github.com/lmontand>`__ for this massive effort!)
* Upgrade code health internally: added isort and expanded flake8 checks.
  (thanks to `nicpayne713 <https://github.com/nicpayne713>`__ and `orsinium <https://github.com/orsinium>`__!)
* Correctly handle preconditions with recursive calls to the contracted function.
  (see `this test <https://github.com/pschanely/CrossHair/commit/c424a0b7060cc22d4afc6c9ffa9cc4ea49bc330d#diff-224c946e97220722461766d8cdb828c3b57945c8f435a572e06bc8f00bb23637>`__)
* Fix symbolic ``str.capitalize()`` behavior in python 3.7.
* CrossHair now has datetime support that doesn't destructively modify the system's
  datetime module.
  (fixes `#159 <https://github.com/pschanely/CrossHair/issues/159>`__)


Version 0.0.22
--------------

* Added a new `specs_complete` directive: use this to let functions
  return any value confirming to their contract.
  This can be useful for
  (`ensuring you don't depend on implementation details <https://crosshair.readthedocs.io/en/latest/case_studies.html#contractual-semver>`__).
* Fix formatting symbolic enums as decimals.
* Use comparisons to guess types for untyped values.
* Permit writes to /dev/null, allowing imports for pytorch.
  (`see #157 <https://github.com/pschanely/CrossHair/issues/157>`__)
* Resolve types dependent on TYPE_CHECKING guards in more cases.
  (`see #158 <https://github.com/pschanely/CrossHair/issues/158>`__)
* Made various diagnosic improvements for ``-v`` output.
* Mix up the message-of-the-day when exiting ``crosshair watch``.
* Implemented minor performance and search heuristic improvements.


Version 0.0.21
--------------

* Add support for memoryview.
  (`see #153 <https://github.com/pschanely/CrossHair/issues/153>`__)
* Use pure-python code for
  `Cython <https://cython.org/>`__
  modules that distribute it.
  This enables symbolic reasoning for modules like
  `Pydantic <https://pydantic-docs.helpmanual.io/>`__
  that include both pure and binary versions.
* Add path search hueristics to bias for code coverage.
* Fix bug in newline detection for ``str.splitlines``.
* Fix bug for title-case characters in ``str.capitalize``.
* Correctly model when ``isinstance``/``issubclass`` over symbolics raise exceptions.
* Completed Python 3.10 support.


Version 0.0.20
--------------

* Complete symbolic support for all string methods!
  (`see #39 <https://github.com/pschanely/CrossHair/issues/39>`__)
* Complete symbolic support JSON encode and decode!
* Add symbolic support for ascii, latin-1, and utf8 encode and decode.
* Add symbolic support for StringIO.
* Fix bugs in string comparisons, re.finditer, isinstance, delete-by-slice.
* Add symbolic support for set comprehensions.
* Add minor optimizations for tracing and repeated slicing.
* Skip copies for uncopy-able arguments
  (`see #146 <https://github.com/pschanely/CrossHair/issues/146>`__)
* Fix bug for special cases when ``__new__`` should be called without ``__init__``


Version 0.0.19
--------------

* Completed full symbolic regex support!

  * The remaining features were non-greedy matching (``.*?``),
    word boundaries (``\b``),
    and negated sets (``[^abc]``).

* Fixed crash on clean installation which expected Deal to be installed - that
  dependency is now fully optional.
  (`issue <https://github.com/pschanely/CrossHair/issues/132>`__)
* Avoid crash when ``crosshair watch`` has been running for a while on trivial cases.
  (`issue <https://github.com/pschanely/CrossHair/issues/131>`__)
* Add symbolic support for f-strings.
* Add symbolic support for dictionary comprehensions with symbolic keys.


Version 0.0.18
--------------

* Add support for counterexamples in full unicode!
  (previously, we'd only find counterexamples in latin-1)
* Add support for checking Deal contracts!
  (:ref:`details <analysis_kind_deal>`)
* Add fixes for
  `collections.deque <https://github.com/pschanely/CrossHair/commit/7df7f86531ba0fbc9a0f3658bee3621951a2099b>`__,
  `float rounding false-positives <https://github.com/pschanely/CrossHair/commit/28217d157be93cfcd445fb50d2955dd7366615b9>`__,
  `dict.pop <https://github.com/pschanely/CrossHair/commit/d8e153d3762a18727d55cbdc524309e9b7f22d12>`__, and
  `nondeterminism detection <https://github.com/pschanely/CrossHair/commit/4f3f9afbeb8b20723c2b623d705326cfcde4f6fe>`__.
* Give
  `reproducible failures <https://github.com/pschanely/CrossHair/commit/3ea61be9e5d2da4adc563e65db8edc391601acea>`__
  for code involving random number generation.
* Add symbolic support for string predicates:
  isalpha, isspace, isascii, isdecimal, isdigit, islower, isnumeric, isprintable,
  isalnum, and istitle.
* Expand symbolic regex support: search, sub, subn, finditer, re.MULTILINE,
  lookahead/lookbehind, and lastindex/lastgroup.


Version 0.0.17
--------------

* Add support for checking Hypothesis tests!
  (:ref:`details <analysis_kind_hypothesis>`)
* **Important**: The ``--analysis_kind=assert`` option is no longer enabled by default.
  (it was spuriously detecting functions for analysis too regularly)
  Enable assert-mode explicitly on the command line if you use CrossHair this way.
* Support the ``analysis_kind`` option in code comment "directives."
* Add some minimal symbolic support for the standard library ``array`` module.
* Add symbolic support for ``bytearray``.
* Expand symbolic support for ord(), chr(), and integer round().
* Expand symbolic support for some bitwise operations and ``int.bit_length``.


Version 0.0.16
--------------

* Add new ``crosshair cover`` command.
  (`details <https://crosshair.readthedocs.io/en/latest/cover.html>`__)
* Implement and document CrossHair's plugin system.
  (`details <https://crosshair.readthedocs.io/en/latest/plugins.html>`__)
* 3rd party Cython modules sometimes include both binary and pure versions of the code.
  Now CrossHair can access the pure Python code in such distributions, allowing it to
  symbolically execute them.
* Add symbolic support for integer and float parsing.
* Add symbolic support for indexing into concrete dictionaries with symbolic keys.
* Add regex support for the whitespace ("\\s") class.
  (regex support is still ASCII-only right now though)
* Miscellaneous fixes: string indexing, numeric promotions, named regex groups


Version 0.0.15
--------------

* Fix regression for ``watch`` command, which crashed when watched files have a syntax
  error.
* Fix ``watch`` command to consistently detect when files are deleted.
* `Expand <https://github.com/pschanely/CrossHair/issues/112>`__ symbolic handling for
  some string containment use cases.
* Refactored tracing intercept logic to support arbitrary opcode interceptions
  (will unlock new symbolic strategies)


Version 0.0.14
--------------

* The type() function is now patched (it no longer reveals symbolic types).
* Completed Python 3.9 support.
* Refined (make less magical) and documented custom class suggestions.
* Fixed out-of-bounds slicing in certain cases.
* Fixed regression breaking check by class name.
* Fixed crash on "watch ." and an excessive auditwall block on os.walk.
* Fixed issue targeting by line number.
* Fixed error on no command line arguments.


Version 0.0.13
--------------

* Further simplification of ``crosshair watch`` output for broader terminal support.


Version 0.0.12
--------------

* Use simpler ``crosshair watch`` screen clearing mechanism for terminals like Thonny's.
* Several string methods can now be reasoned about symbolically: split, find, replace,
  index, partition, count, and more.
  (thanks `Rik-de-Kort <https://github.com/Rik-de-Kort>`_!)
* Fixed various bugs, including a few specific to icontract analysis.
* Modestly increased regex cases that CrossHair handles. (including named groups!)


Version 0.0.11
--------------

* `Enable <https://github.com/pschanely/CrossHair/issues/84>`__
  analysis when only preconditions exist. (this is useful if you just want to catch
  exceptions!)
* Added ``--report_verbose`` option to customize whether you get verbose multi-line
  counterexample reports or the single-line, machine-readable reporting.
  (`command help <https://crosshair.readthedocs.io/en/latest/command-line_interface.html#check>`__)
* Added workaround for missing ``crosshair watch`` output in the PyCharm terminal.
* Assorted bug fixes:
  `1 <https://github.com/pschanely/CrossHair/pull/90>`__,
  `2 <https://github.com/pschanely/CrossHair/pull/92>`__,
  `3 <https://github.com/pschanely/CrossHair/commit/95b6dd1bff0ab186ac61c153fc15d231f7020f1c>`__,
  `4 <https://github.com/pschanely/CrossHair/commit/1110d8f81ff967f11fc1439ef4abcf301276f309>`__


Version 0.0.10
--------------

* Added support for checking
  `icontract <https://github.com/Parquery/icontract>`_
  postconditions.
  (`details <https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html#analysis-kind-icontract>`__)
* Added support for checking plain ``assert`` statements.
  (`details <https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html#assert-based-contracts>`__)
* Expanded & refactored the
  `documentation <https://crosshair.readthedocs.io/en/latest/index.html>`__.
  (thanks `mristin <https://github.com/mristin>`_!)
* Advanced internal code standards: black, mypy, pydocstyle, and more.
  (thanks `mristin <https://github.com/mristin>`_!)
* Added basic protection against dangerous side-effects with ``sys.addaudithook``.
* Analysis can now be targeted by function at line number; e.g. ``crosshair check foo.py:42``
* Modules and functions may include a directive comment like ``# crosshair: on`` or
  ``# crosshair: off`` to customize targeting.
* Realization heuristics enable solutions for some use cases
  `like this <https://github.com/pschanely/CrossHair/blob/b47505e7957e5f22a05dd6a785429b6b3f408a68/crosshair/libimpl/builtinslib_test.py#L353>`__
  that are challenging for Z3.
* Enable symbolic reasoning about getattr and friends.
  (`example <hhttps://github.com/pschanely/CrossHair/blob/main/crosshair/examples/PEP316/bugs_detected/getattr_magic.py>`__)
* Fixes or improvements related to:

  * builtin tolerance for symbolic values
  * User-defined class proxy generation
  * Classmethods on int & float.
  * Floordiv and mod operators
  * ``list.index()`` and list ordering
  * The ``Final[]`` typing annotation
  * xor operations over sets


Version 0.0.9
-------------

* Introduce :ref:`the diffbehavior command <diffbehavior>` which finds
  inputs that distinguish the behavior of two functions.
* Upgrade to the latest release of Z3 (4.8.9.0)
* Fix `an installation error on Windows <issue_41_>`_.
* Fix a variety of other bugs.

.. _issue_41: https://github.com/pschanely/CrossHair/issues/41
