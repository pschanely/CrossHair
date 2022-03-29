.. _diffbehavior:

*************************
The Details: diffbehavior
*************************

Are these two functions equivalent?::

    # foo.py
    from typing import List

    def cut1(a: List[int], i: int) -> None:
      a[i:i+1] = []

    def cut2(a: List[int], i: int) -> None:
      a[:] = a[:i] + a[i+1:]

Almost! But not quite.

CrossHair's ``diffbehavior`` command can help you find out::

    $ crosshair diffbehavior foo.cut1 foo.cut2

    Given: (a=[9, 0], i=-1),
      foo.cut1 : after execution a=[9, 0]
      foo.cut2 : after execution a=[9, 9, 0]

How do I try it?
================

.. code-block::

    $ pip install crosshair-tool
    $ crosshair diffbehavior <module>.<function> <module>.<function>


``diffbehavior``
================

.. code-block::

    crosshair diffbehavior --help

.. Help starts: crosshair diffbehavior --help
.. code-block:: text

    usage: crosshair diffbehavior [-h] [--verbose] [--per_path_timeout FLOAT]
                                  [--per_condition_timeout FLOAT]
                                  [--contract_file CONTRACT_FILE [CONTRACT_FILE ...]]
                                  FUNCTION1 FUNCTION2

    Find differences in the behavior of two functions.
    See https://crosshair.readthedocs.io/en/latest/diff_behavior.html

    positional arguments:
      FUNCTION1             first fully-qualified function to compare (e.g. "mymodule.myfunc")
      FUNCTION2             second fully-qualified function to compare

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --per_path_timeout FLOAT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout FLOAT
                            Maximum seconds to spend checking execution paths for one condition
      --contract_file CONTRACT_FILE [CONTRACT_FILE ...]
                            Register contracts, running the given python file

.. Help ends: crosshair diffbehavior --help


diffbehavior your own code changes
======================================

Use ``git worktree`` to create an unmodified source tree, and then use
``crosshair diffbehavior`` to compare your local version to head.

.. code-block::

    # Let's say we edit the clean() function in foo.py

    # Step 1: Create an unmodified source tree under a directory named "clean":
    $ git worktree add --detach clean

    # Step 2: Have CrossHair try to detect a difference:
    $ crosshair diffbehavior foo.cut clean.foo.cut

    # Step 3: Remove the "clean" directory when you're done:
    $ git worktree remove clean

.. _an-example-shell-function:

An example shell function
=========================

If you find yourself doing this often, make a function or script.
For example, you might put this function in your ``~/.bashrc`` file:

.. code-block:: bash

    diffbehavior() {
        git worktree add --detach _clean || exit 1
        crosshair diffbehavior "$1" "_clean.$@"
        git worktree remove _clean
    }

Then, you can diff your uncommitted changes very easily:

.. code-block:: bash

    $ diffbehavior foo.cut
    ...

Refactoring? Use diffbehavior to make sure it's safe.
=========================================================

Say we start with this:

.. code-block:: python

    # foo.py
    def longest_str(items: List[str]) -> str:
      longest = ''
      for item in items:
        if len(item) > len(longest):
          longest = item
      return longest


... and change it to this:

.. code-block:: python

    def longest_str(items: List[str]) -> str:
      return max(items,
                 key=lambda item: len(item),
                 default='')

We can use :ref:`the shell function above <an-example-shell-function>` to help
make sure the code doesn't operate differently:

.. code-block:: bash

    $ diffbehavior foo.longest_str
    No differences found. (attempted 15 iterations)
    Consider trying longer with: --per_condition_timeout=<seconds>

Developing new features or fixing bugs? ``diffbehavior`` finds inputs to test.
==============================================================================

Say we start with this:

.. code-block:: python

    def isack(s: str) -> bool:
        if s in ('y', 'yes'):
            return True
        return False

... and change it to this:

.. code-block:: python

    def isack(s: str) -> bool:
        if s in ('y', 'yes', 'Y', 'YES'):
            return True
        if s in ('n', 'no', 'N', 'NO'):
            return False
        raise ValueError('invalid ack')

We can use :ref:`the shell function above <an-example-shell-function>` to find
useful inputs for testing:

.. code-block::

    $ diffbehavior foo.isack
    Given: (s='\x00'),
             foo.isack : returns False
      _clean.foo.isack : raises ValueError('invalid ack')
    Given: (s='YES'),
             foo.isack : returns False
      _clean.foo.isack : returns True

CrossHair reports examples in order of added coverage, descending, so consider
writing your unit tests using such inputs, from the top-down.

But don't do it blindly! CrossHair doesn't always give pleasant examples;
instead of using ``'\x00'``, you should just use ``'a'`` to cover the same
logic.

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
* Be aware that the absence of an example difference does not guarantee that
  the functions are equivalent.
* CrossHair likely won't be able to detect differences in complex code. Target
  it at the smallest piece of logic possible.
* Your arguments must have proper `type annotations`_.
* Your arguments have to be deep-copyable and equality-comparable. (this is so
  that we can detect code that mutates them)
* CrossHair is supported only on Python 3.7+ and only on CPython (the most
  common Python implementation).
* Only deterministic behavior can be analyzed.
  (your code always does the same thing when starting with the same values)
* Be careful: CrossHair will actually run your code and may apply any arguments
  to it.

.. _type annotations: https://www.python.org/dev/peps/pep-0484/

Credits
=======

The diffbehavior command was inspired by `Hillel Wayne`_'s post about
`cross-branch testing`_!

.. _Hillel Wayne: http://hillelwayne.com/
.. _cross-branch testing: https://buttondown.email/hillelwayne/archive/cross-branch-testing/
