***************************
Why Should I Use CrossHair?
***************************

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

**Inter-procedural analysis.**
CrossHair (1) validates the pre-conditions of called functions and
(2) uses post-conditions of called functions to help it prove post-conditions
in the caller.

.. image:: zipped_pairs.png
    :width: 488
    :height: 214
    :alt: Image showing CrossHair caller and callee

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
CrossHair is a nice companion to `mypy`_].
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
