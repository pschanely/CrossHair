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

.. note::

    The inputs that CrossHair generates may not be very readable or minimal.
    However, you may find that increasing the `--per_condition_timeout` yields smaller inputs.


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

    usage: crosshair cover [-h] [--verbose] [--extra_plugin FILE [FILE ...]]
                           [--unblock EVENT [EVENT ...]]
                           [--example_output_format FORMAT] [--coverage_type TYPE]
                           [--max_uninteresting_iterations MAX_UNINTERESTING_ITERATIONS]
                           [--per_path_timeout FLOAT]
                           [--per_condition_timeout FLOAT]
                           TARGET [TARGET ...]

    Generates inputs to a function, hopefully getting good line, branch,
    and path coverage.
    See https://crosshair.readthedocs.io/en/latest/cover.html

    positional arguments:
      TARGET                A fully qualified module, class, or function, or
                            a directory (which will be recursively analyzed), or
                            a file path with an optional ":<line-number>" suffix.
                            See https://crosshair.readthedocs.io/en/latest/contracts.html#targeting

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --extra_plugin FILE [FILE ...]
                            Plugin file(s) you wish to use during the current execution
      --unblock EVENT [EVENT ...]
                            Allow specific side-effects. See the list of audit events at:
                            https://docs.python.org/3/library/audit_events.html
                            You may specify colon-delimited event arguments to narrow the unblock, e.g.:
                                --unblock subprocess.Popen:echo
                            Finally, `--unblock EVERYTHING` will disable all side-effect detection.
      --example_output_format FORMAT
                            Determines how to output examples.
                                eval_expression     : [default] Output examples as expressions,
                                                      suitable for eval()
                                arg_dictionary      : Output arguments as repr'd, ordered
                                                      dictionaries
                                pytest              : Output examples as stub pytest tests
                                argument_dictionary : Deprecated
      --coverage_type TYPE  Determines what kind of coverage to achieve.
                                opcode : [default] Cover as many opcodes of the function as
                                         possible. This is similar to "branch" coverage.
                                path   : Cover any possible execution path.
                                         There will usually be an infinite number of paths (e.g.
                                         loops are effectively unrolled). Use
                                         max_uninteresting_iterations and/or per_condition_timeout
                                         to bound results.
                                         Many path decisions are internal to CrossHair, so you may
                                         see more duplicative-ness in the output than you'd expect.
      --max_uninteresting_iterations MAX_UNINTERESTING_ITERATIONS
                            Maximum number of consecutive iterations to run without making
                            significant progress in exploring the codebase.
                            (by default, 5 iterations, unless --per_condition_timeout is set)

                            This option can be more useful than --per_condition_timeout
                            because the amount of time invested will scale with the complexity
                            of the code under analysis.

                            Use a small integer (3-5) for fast but weak analysis.
                            Values in the hundreds or thousands may be appropriate if you
                            intend to run CrossHair for hours.
      --per_path_timeout FLOAT
                            Maximum seconds to spend checking one execution path.
                            If unspecified:
                            1. CrossHair will timeout each path at the square root of
                               `--per_condition_timeout`, if specified.
                            3. Otherwise, it will timeout each path at a number of seconds
                               equal to `--max_uninteresting_iterations`, unless it is
                               explicitly set to zero.
                               (NOTE: `--max_uninteresting_iterations` is 5 by default)
                            2. Otherwise, it will not use any per-path timeout.
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

* CrossHair likely won't be able to fully explore complex code.
* Your arguments must have proper `type annotations`_.
* Your arguments have to be deep-copyable and equality-comparable.
* Arguments that are (or contain) instances of user-defined classes must meet certain expectations;
  see :ref:`Hints for Your Classes <hints_for_your_classes>`.
* Tests are generated in the form ``assert <invocation> == <repr of return>``. Consequently, all return values
  will need to be equality comparable and have reprs that faithfully reconstruct any object state.
* CrossHair is supported only on Python 3.8+ and only on CPython (the most
  common Python implementation).
* Only deterministic behavior can be analyzed.
  (your code always does the same thing when starting with the same values)
* Be careful: CrossHair will actually run your code and may apply any arguments
  to it.

.. _type annotations: https://www.python.org/dev/peps/pep-0484/
