************
Related Work
************

**Related Topics and Tools:**

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
    CrossHair is essentially a concolic testing system.
    This kind of tool sits in a middle ground in between property testing and formal
    methods.

`formal methods`_, `Nagini`_
    CrossHair is good at finding counterexamples, but cannot generally prove that
    properties hold. Formal method tools can help you do that.
    For example, Nagini, is a verifier for Python programs.

    The main downside of full verification is that you'll need to understand a bit
    about how the verifier works and may need to guide it with additional information
    like loop invariants and termination proofs.

`SMT solvers`_
    SMT solvers power many of the tools listed here. CrossHair uses `Z3`_.

`angr`_, `klee`_
    Symbolic execution of **binary** code.
    Unlike these tools, CrossHair models the semantics of Python directly.

`PyExZ3`_, `pySim`_, `PEF`_
    These projects take approaches are very similar to CrossHair, and are in varying
    states of completeness.
    CrossHair is generally more prescriptive or product-like than these tools.


**Work Involving CrossHair:**

    * Andrea Veneziano & Samuel Chassot. 2022.
      `SVSHI: Secure and Verified Smart Home Infrastructure <https://arxiv.org/pdf/2206.11786>`__.
      (see also the `github repo <https://github.com/dslab-epfl/svshi>`__ and in particular the
      `docs on verification <https://github.com/dslab-epfl/svshi/blob/main/src/documentation/documentation.md#433-verification>`__)
    * Loïc Montandon. 2022.
      `Exhaustive symbolic execution engine for verifying Python programs <https://github.com/dslab-epfl/svshi/blob/main/src/documentation/reports/Exhaustive_Crosshair_and_external_functions%20-%20Loic%20Montandon.pdf>`__.
    * Aymeri Servanin. 2022.
      `Extending the verification of SVSHI applications with time-sensitive constraints <https://github.com/dslab-epfl/svshi/blob/main/src/documentation/reports/Thesis%20Improving%20SVSHI-s%20verification%20-%20Aymeri%20Servanin.pdf>`__.
    * Jiyang Zhang, Marko Ristin, Phillip Schanely, Hans Wernher van_de_Venn, Milos Gligoric. 2022.
      `Python-by-Contract Dataset <https://jiyangzhang.github.io/files/ZhangETAL22PyContract.pdf>`__.
    * Shikha Mody, Bradley Mont, Jivan Gubbi, & Brendon Ng. 2022.
      `Semantic Merge Conflict Detection <https://github.com/shikham-8/CS230-TIM-Improves-Merging/blob/main/CS_230_TIM_Report.pdf>`__.
      (see also the `github repo <https://github.com/shikham-8/CS230-TIM-Improves-Merging>`__)


.. _dependent types: https://en.wikipedia.org/wiki/Dependent_type
.. _refinement types: https://en.wikipedia.org/wiki/Refinement_type
.. _design by contract: https://en.wikipedia.org/wiki/Design_by_contract
.. _fuzz testing: https://en.wikipedia.org/wiki/Fuzzing
.. _QuickCheck: https://en.wikipedia.org/wiki/QuickCheck
.. _property testing: https://en.wikipedia.org/wiki/Property_testing
.. _Hypothesis: https://hypothesis.readthedocs.io/
.. _concolic testing: https://en.wikipedia.org/wiki/Concolic_testing
.. _formal methods: https://en.wikipedia.org/wiki/Formal_methods
.. _Nagini: https://github.com/marcoeilers/nagini
.. _SMT solvers: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories
.. _Z3: https://github.com/Z3Prover/z3
.. _angr: https://angr.io
.. _klee: https://klee.github.io/
.. _PyExZ3: https://github.com/thomasjball/PyExZ3
.. _pySim: https://github.com/bannsec/pySym
.. _PEF: https://git.cs.famaf.unc.edu.ar/dbarsotti/pef