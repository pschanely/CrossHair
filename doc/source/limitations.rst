***********
Limitations
***********

A (wildly incomplete) list of present limitations.
Some of these will be lifted over time (your help is welcome!);
some may never be lifted.

* Be aware that the absence of a counterexample does not guarantee that
  the property holds.
* Symbolic values are implemented as Python proxy values.
  CrossHair patches the system to maintain a good illusion, but the illusion is not
  complete. For example, code that cares about the identity values (x is y) may not be
  correctly analyzed.
* Only function and class definitions at the top level are analyzed
  (*i.e.* not when nested inside other functions or classes).
* Only deterministic behavior can be analyzed
  (*i.e.* your code always does the same thing when starting
  with the same values).
  * CrossHair may produce a ``NotDeterministic`` error when it detects this.

* Be careful: CrossHair will actually run your code and may apply any arguments
  to it.

  * CrossHair puts some protections in place (in 3.8+ only, via ``sys.addaudithook``) to
    prevent disk and network access, but this protection is not perfect (notably, it
    will not prevent actions taken by C-based modules)

* Consuming values of an iterator or a generator in a pre- or post-condition
  will produce `unexpected behavior`_.
* SMT solvers have very different perspectives on hard problems and
  easy problems than humans.

  * Be prepared to be surprised both by what CrossHair can tell you,
    and what it cannot.

.. _unexpected behavior: https://github.com/pschanely/CrossHair/issues/9
