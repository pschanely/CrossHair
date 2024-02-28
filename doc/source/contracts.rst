.. _contracts:

**********************
The Details: Contracts
**********************

Learn more about contract-checking below!


Contract Syntaxes
=================

CrossHair can check many different kinds of contracts; choose one that fits you best:

+----------------------------------------------+--------------------------------------------------------------------------+
| :ref:`asserts <analysis_kind_asserts>`       | Use regular Python assert statements. That's it.                         |
|                                              |                                                                          |
+----------------------------------------------+--------------------------------------------------------------------------+
| :ref:`PEP 316 <analysis_kind_pep316>`        | Docstring-based contracts.                                               |
|                                              |                                                                          |
+----------------------------------------------+--------------------------------------------------------------------------+
| :ref:`icontract <analysis_kind_icontract>`   | Some 3rd party contract libraries.                                       |
|                                              |                                                                          |
+----------------------------------------------+ These contracts are attached to your functions with decorators.          |
| :ref:`deal <analysis_kind_deal>`             |                                                                          |
|                                              |                                                                          |
+----------------------------------------------+--------------------------------------------------------------------------+
| :ref:`Hypothesis <analysis_kind_hypothesis>` | hypothesis property-based tests can also be checked.                     |
|                                              |                                                                          |
|                                              | (even though they aren't "contracts," strictly speaking)                 |
+----------------------------------------------+--------------------------------------------------------------------------+


Targeting
=========

There are many different ways to specify what CrossHair should check:

* Directories. e.g. ``crosshair check mypkg/``
* Files. e.g. ``crosshair check mypkg/foo.py``
* File and line number. e.g. ``crosshair check mypkg/foo.py:23``
* Modules. e.g. ``crosshair check mypkg.foo``
* Classes. e.g. crosshair ``check mypkg.foo.MyClass``
* Functions or methods. e.g. crosshair ``check mypkg.foo.MyClass.my_method``


.. _contract_configuration:

Configuration
=============

In addition the the targeting and options specified on the
:ref:`Watch <contract_watch>`
and
:ref:`Check <contract_check>`
command lines, you can customize CrossHair's analysis with special
comments ("directives") in your code, like this::

    # crosshair: off

    def grow(age: int):
        # crosshair: on
        # crosshair: analysis_kind=asserts
        assert age >= 0
        ...

Directives may appear in the body of a function, at the top level of a module,
or in the ``__init__.py`` file of a package.

Notably, you may want to specify your contract syntax
(``# crosshair: analysis_kind=icontract``)
in a top-level ``__init__.py`` file.

Lower level directives take precedence over higher level directives.

These are the most commonly used directives:

* ``# crosshair: off`` - disable contract checking.
* ``# crosshair: on`` - re-enable contract checking.
* ``# crosshair: analysis_kind=<KIND>`` - set the kind of contract to check


.. note::
    CrossHair only evaluates code that is **reachable by running some function with a
    contract**.

    Even if a function is targeted, it isn't analyzed unless it has at least one
    pre- or post-condition.
    It is common to set a trivial post-condition of "True"  on a function to tell
    CrossHair it is a valid entry point for analysis.


.. _contract_watch:

Watch
=====

The watch command continuously looks for contract counterexamples.
Type Ctrl-C to stop this command.

.. Help starts: crosshair watch --help
.. code-block:: text

    usage: crosshair watch [-h] [--verbose]
                           [--extra_plugin EXTRA_PLUGIN [EXTRA_PLUGIN ...]]
                           [--analysis_kind KIND]
                           TARGET [TARGET ...]

    The watch command continuously looks for contract counterexamples.
    Type Ctrl-C to stop this command.

    positional arguments:
      TARGET                File or directory to watch. Directories will be recursively analyzed.
                            See https://crosshair.readthedocs.io/en/latest/contracts.html#targeting

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --extra_plugin EXTRA_PLUGIN [EXTRA_PLUGIN ...]
                            Plugin file(s) you wish to use during the current execution
      --analysis_kind KIND  Kind of contract to check.
                            By default, the PEP316, deal, and icontract kinds are all checked.
                            Multiple kinds (comma-separated) may be given.
                            See https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html
                                asserts    : check assert statements
                                PEP316     : check PEP316 contracts (docstring-based)
                                icontract  : check icontract contracts (decorator-based)
                                deal       : check deal contracts (decorator-based)
                                hypothesis : check hypothesis tests

.. Help ends: crosshair watch --help


.. _contract_check:

Check
=====

The check command looks for counterexamples that break contracts.
It is more customizable than ``watch`` and produces machine-readable output.

.. Help starts: crosshair check --help
.. code-block:: text

    usage: crosshair check [-h] [--verbose]
                           [--extra_plugin EXTRA_PLUGIN [EXTRA_PLUGIN ...]]
                           [--report_all] [--report_verbose]
                           [--max_uninteresting_iterations MAX_UNINTERESTING_ITERATIONS]
                           [--per_path_timeout FLOAT]
                           [--per_condition_timeout FLOAT] [--analysis_kind KIND]
                           TARGET [TARGET ...]

    The check command looks for counterexamples that break contracts.

    It outputs machine-readable messages in this format on stdout:
        <filename>:<line number>: error: <error message>

    It exits with one of the following codes:
        0 : No counterexamples are found
        1 : Counterexample(s) have been found
        2 : Other error

    positional arguments:
      TARGET                A fully qualified module, class, or function, or
                            a directory (which will be recursively analyzed), or
                            a file path with an optional ":<line-number>" suffix.
                            See https://crosshair.readthedocs.io/en/latest/contracts.html#targeting

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --extra_plugin EXTRA_PLUGIN [EXTRA_PLUGIN ...]
                            Plugin file(s) you wish to use during the current execution
      --report_all          Output analysis results for all postconditions (not just failing ones)
      --report_verbose      Output context and stack traces for counterexamples
      --max_uninteresting_iterations MAX_UNINTERESTING_ITERATIONS
                            Maximum number of consecutive iterations to run without making
                            significant progress in exploring the codebase.

                            This option can be more useful than --per_condition_timeout
                            because the amount of time invested will scale with the complexity
                            of the code under analysis.

                            Use a small integer (3-5) for fast but weak analysis.
                            Values in the hundreds or thousands may be appropriate if you intend to
                            run CrossHair for hours.
      --per_path_timeout FLOAT
                            Maximum seconds to spend checking one execution path.
                            If unspecified, CrossHair will timeout each path:
                            1. At the square root of `--per_condition_timeout`, if specified.
                            2. Otherwise, at a number of seconds equal to
                               `--max_uninteresting_iterations`, if specified.
                            3. Otherwise, there will be no per-path timeout.
      --per_condition_timeout FLOAT
                            Maximum seconds to spend checking execution paths for one condition
      --analysis_kind KIND  Kind of contract to check.
                            By default, the PEP316, deal, and icontract kinds are all checked.
                            Multiple kinds (comma-separated) may be given.
                            See https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html
                                asserts    : check assert statements
                                PEP316     : check PEP316 contracts (docstring-based)
                                icontract  : check icontract contracts (decorator-based)
                                deal       : check deal contracts (decorator-based)
                                hypothesis : check hypothesis tests

.. Help ends: crosshair check --help


Example Uses
============

These examples use the :ref:`PEP 316 <analysis_kind_pep316>` format,
but the motivation applies to :ref:`all contract kinds <Kinds of Contracts>`.

**More precision.**
Commonly, we care about more than just the type.
Is it really any integer, or is it a **positive** integer?
Is it any list, or does it have to be a non-empty list?
CrossHair gives you that precision:

.. image:: average.png
    :width: 387
    :height: 111
    :alt: Image showing an average function

**Verify across all implementations.**
Contracts are particularly helpful when applied to base classes and interfaces:
all implementations will be verified against them:

.. image:: chess_pieces.png
    :width: 545
    :height: 336
    :alt: Image showing CrossHair contract and inheritance

**Catch errors.**
Setting a trivial post-condition of ``True`` is enough to enable analysis,
which will find exceptions like index out-of-bounds errors:

.. image:: index_bounds.gif
    :width: 610
    :height: 192
    :alt: Image showing CrossHair contract and IndexError

**Support your type checker.**
CrossHair is a nice companion to `mypy`_.
Assert statements divide work between the two systems:

.. image:: pair_with_mypy.png
    :width: 512
    :height: 372
    :alt: Image showing mypy and CrossHair together

.. _mypy: http://mypy-lang.org/

**Optimize with Confidence.**
Using post-conditions, CrossHair ensures that optimized code continues to
behave like equivalent naive code:

.. image:: csv_first_column.png
    :width: 502
    :height: 198
    :alt: Image showing the equivalence of optimized an unoptimized code

**More Examples**
You can find examples in the `examples/`_ directory and
try CrossHair in your browser at `crosshair-web.org`_.

.. _examples/: https://github.com/pschanely/CrossHair/tree/main/crosshair/examples
.. _crosshair-web.org: https://crosshair-web.org


Is CrossHair executing my code?
===============================

CrossHair **does truly execute your contracted functions**,
but it supplies special symbolic arguments,
and intercepts many of the usual Python behaviors while doing so.

It may or may not execute your preconditions and postconditions.
It'll usually execute the code of subroutines as well, but doesn't always, and may
execute that logic out-of-order.
Mostly, you don't need to worry about these details, but some of these effects may
become visible if your code calls ``print()``, for instance.
(note further that printing symbolic values will force them to take on concrete values
and will hamper CrossHair's ability to effectively analyze your code!)

Because of the wide variety of things CrossHair might do, never target code that can
directly or indirectly cause side-effects.
CrossHair puts some protections in place (via ``sys.addaudithook``) to prevent disk
and network access, but this protection is not perfect. (notably, it will not
prevent actions taken by C-based modules)
