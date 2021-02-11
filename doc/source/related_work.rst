************
Related Work
************

`dependent types`_, `refinement types`_
    CrossHair aims to provide many of the same capabilities as these
    advanced type systems.
    CrossHair is easier to learn (because it is just python), but is
    incomplete.
    It can't always tell you whether a condition holds.

`design by contract`_
    Unlike most tools for contracts, CrossHair attempts to verify
    pre-conditions and post-conditions before you run them.

`fuzz testing`_, `QuickCheck`_, `property testing`_, `Hypothesis`_
    CrossHair has many of the same goals as these tools.
    However, CrossHair uses an SMT solver to find inputs rather than
    the (typically) randomized approaches that these tools use.

`concolic testing`_
    State-of-the-art fuzz testers employ SMT solvers in a similar fashion
    as CrossHair.

`SMT solvers`_
    SMT solvers power many of the tools in this table. CrossHair uses `Z3`_.

`angr`_, `klee`_
    Symbolic execution of **binary** code.
    Unlike these tools, CrossHair models the semantics of Python directly.

`PyExZ3`_, `pySim`_, `PEF`_
    Take approaches that are very similar to CrossHair, in various states
    of completeness.
    CrossHair is generally more prescriptive or product-like than
    these tools.

.. _dependent types: https://en.wikipedia.org/wiki/Dependent_type
.. _refinement types: https://en.wikipedia.org/wiki/Refinement_type
.. _design by contract: https://en.wikipedia.org/wiki/Design_by_contract
.. _fuzz testing: https://en.wikipedia.org/wiki/Fuzzing
.. _QuickCheck: https://en.wikipedia.org/wiki/QuickCheck
.. _property testing: https://en.wikipedia.org/wiki/Property_testing
.. _Hypothesis: https://hypothesis.readthedocs.io/
.. _concolic testing: https://en.wikipedia.org/wiki/Concolic_testing
.. _SMT solvers: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories
.. _Z3: https://github.com/Z3Prover/z3
.. _angr: https://angr.io
.. _klee: https://klee.github.io/
.. _PyExZ3: https://github.com/thomasjball/PyExZ3
.. _pySim: https://github.com/bannsec/pySym
.. _PEF: https://git.cs.famaf.unc.edu.ar/dbarsotti/pef