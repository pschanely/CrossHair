.. _cover:

*************************
The Details: cover
*************************

CrossHair's ``cover`` command finds inputs to your function that get coverage over it::

    $ cat foo.py
    from typing import List, Optional
    def average(nums: List[float], default:Optional[float] = None) -> float:
        if len(nums) == 0:
            if default is None:
                raise ValueError
            return default
        return sum(nums) / len(nums)

    $ crosshair cover foo.average
    average([0.0, 0.0], None)
    average([], 0.0)
    average([], None)

CrossHair reports examples in order of added (opcode-level) coverage, descending.

You can even use the ``--example_output_format=pytest`` option to jumpstart your unit
tests!::

    $ crosshair cover --example_output_format=pytest foo.average
    import pytest
    from foo import average

    def test_average():
        assert average([0.0, 0.0], None) == 0.0

    def test_average_2():
        assert average([], 0.0) == 0.0

    def test_average_3():
        with pytest.raises(ValueError):
            average([], None)

But don't do this blindly!
CrossHair only reports what your code does, not what it is supposed to do.
Also note that CrossHair example data may not be minimal or very readable.


How do I try it?
================

.. code-block::

    $ pip install crosshair-tool
    $ crosshair cover <module>.<function>


``cover``
=========

.. code-block::

    crosshair cover --help

.. Help starts: crosshair cover --help
.. code-block:: text

    usage: crosshair cover [-h] [--verbose]
                           [--extra_plugin EXTRA_PLUGIN [EXTRA_PLUGIN ...]]
                           [--example_output_format FORMAT] [--coverage_type TYPE]
                           [--max_uninteresting_iterations MAX_UNINTERESTING_ITERATIONS]
                           [--per_path_timeout FLOAT]
                           [--per_condition_timeout FLOAT]
                           TARGET [TARGET ...]

    Generates inputs to a function, hopefully getting good line, branch, and path
    coverage.
    See https://crosshair.readthedocs.io/en/latest/cover.html

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
      --example_output_format FORMAT
                            Determines how to output examples.
                                eval_expression     : [default] Output examples as expressions, suitable for
                                                      eval()
                                arg_dictionary      : Output arguments as repr'd, ordered dictionaries
                                pytest              : Output examples as stub pytest tests
                                argument_dictionary : Deprecated
      --coverage_type TYPE  Determines what kind of coverage to achieve.
                                opcode : [default] Cover as many opcodes of the function as possible.
                                         This is similar to "branch" coverage.
                                path   : Cover any possible execution path.
                                         There will usually be an infinite number of paths (e.g. loops are
                                         effectively unrolled). Use max_uninteresting_iterations and/or
                                         per_condition_timeout to bound results.
                                         Many path decisions are internal to CrossHair, so you may see more
                                         duplicative-ness in the output than you'd expect.
      --max_uninteresting_iterations MAX_UNINTERESTING_ITERATIONS
                            Maximum number of consecutive iterations to run without making
                            significant progress in exploring the codebase.

                            This option can be useful than --per_condition_timeout
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

.. Help ends: crosshair cover --help


How does this work?
===================

CrossHair uses an `SMT solver`_ (a kind of theorem prover) to explore execution
paths and look for arguments.
It uses the same engine as the ``crosshair check`` and ``crosshair watch``
commands which check code contracts.

.. _SMT solver: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories


Caveats
=======

* This feature, as well as CrossHair generally, is a work in progress. If you
  are willing to try it out, thank you! Please file bugs or start discussions
  to let us know how it went.
* CrossHair likely won't be able to fully explore complex code.
* Your arguments must have proper `type annotations`_.
* Your arguments have to be deep-copyable and equality-comparable.
* CrossHair is supported only on Python 3.7+ and only on CPython (the most
  common Python implementation).
* Only deterministic behavior can be analyzed.
  (your code always does the same thing when starting with the same values)
* Be careful: CrossHair will actually run your code and may apply any arguments
  to it.

.. _type annotations: https://www.python.org/dev/peps/pep-0484/
