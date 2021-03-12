#########
Changelog
#########

============
Next Version
============

* Nothing yet!

==============
Version 0.0.11
==============

* `Enable <https://github.com/pschanely/CrossHair/issues/84>`_
  analysis when only preconditions exist. (this is useful if you just want to catch
  exceptions!)
* Added ``--report_verbose`` option to customize whether you get verbose multi-line
  counterexample reports or the single-line, machine-readable reporting.
  (`command help <https://crosshair.readthedocs.io/en/latest/command-line_interface.html#check>`_)
* Added workaround for missing ``crosshair watch`` output in the PyCharm terminal.
* Assorted bug fixes:
  `1 <https://github.com/pschanely/CrossHair/pull/90>`_,
  `2 <https://github.com/pschanely/CrossHair/pull/92>`_,
  `3 <https://github.com/pschanely/CrossHair/commit/95b6dd1bff0ab186ac61c153fc15d231f7020f1c>`_,
  `4 <https://github.com/pschanely/CrossHair/commit/1110d8f81ff967f11fc1439ef4abcf301276f309>`_

==============
Version 0.0.10
==============

* Added support for checking
  `icontract <https://github.com/Parquery/icontract>`_
  postconditions. 
  (`details <https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html#analysis-kind-icontract>`_)
* Added support for checking plain ``assert`` statements.
  (`details <https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html#assert-based-contracts>`_)
* Expanded & refactored the 
  `documentation <https://crosshair.readthedocs.io/en/latest/index.html>`_.
  (thanks `mristin <https://github.com/mristin>`_!)
* Advanced internal code standards: black, mypy, pydocstyle, and more.
  (thanks `mristin <https://github.com/mristin>`_!)
* Added basic protection against dangerous side-effects with ``sys.addaudithook``.
* Analysis can now be targeted by function at line number; e.g. ``crosshair check foo.py:42``
* Modules and functions may include a directive comment like ``# crosshair: on`` or
  ``# crosshair: off`` to customize targeting.
* Realization heuristics enable solutions for some use cases
  `like this <https://github.com/pschanely/CrossHair/blob/b47505e7957e5f22a05dd6a785429b6b3f408a68/crosshair/libimpl/builtinslib_test.py#L353>`_
  that are challenging for Z3.
* Enable symbolic reasoning about getattr and friends.
  (`example <https://github.com/pschanely/CrossHair/blob/master/crosshair/examples/PEP316/bugs_detected_fast/getattr_magic.py>`_)
* Fixes or improvements related to:

  * builtin tolerance for symbolic values
  * User-defined class proxy generation
  * Classmethods on int & float.
  * Floordiv and mod operators
  * ``list.index()`` and list ordering
  * The ``Final[]`` typing annotation
  * xor operations over sets


=============
Version 0.0.9
=============

* Introduce :ref:`the diffbehavior command <diffbehavior>` which finds
  inputs that distinguish the behavior of two functions.
* Upgrade to the latest release of Z3 (4.8.9.0)
* Fix `an installation error on Windows <issue_41_>`_.
* Fix a variety of other bugs.

.. _issue_41: https://github.com/pschanely/CrossHair/issues/41
