**********************
Command-line Interface
**********************

Crosshair provides the following commands:

.. code-block::

    crosshair --help

.. Help starts: crosshair --help
.. code-block:: text

    usage: crosshair [-h] {check,watch,diffbehavior} ...

    CrossHair Analysis Tool

    positional arguments:
      {check,watch,diffbehavior}
                            sub-command help
        check               Analyze a file or function
        watch               Continuously watch and analyze a directory
        diffbehavior        Find differences in the behavior of two functions

    optional arguments:
      -h, --help            show this help message and exit

.. Help ends: crosshair --help

``check``
=========

.. code-block::

    crosshair check --help

.. Help starts: crosshair check --help
.. code-block:: text

    usage: crosshair check [-h] [--verbose] [--per_path_timeout FLOAT]
                           [--per_condition_timeout FLOAT] [--report_all]
                           [--analysis_kind KIND]
                           FILE [FILE ...]

    The check command looks for counterexamples that break contracts.

    It outputs machine-readable messages in this format on stdout:
        <filename>:<line number>: error: <error message>

    It exits with one of the following codes:
        0 : No counterexamples are found
        1 : Counterexample(s) have been found
        2 : Other error

    positional arguments:
      FILE                  file/directory or fully qualified module, class, or function

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --per_path_timeout FLOAT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout FLOAT
                            Maximum seconds to spend checking execution paths for one condition
      --report_all          Output analysis results for all postconditions (not just failing ones)
      --analysis_kind KIND  Kind of contract to check. By default, all kinds are checked.
                            See https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html
                                PEP316    : docstring-based contracts
                                icontract : decorator-based contracts
                                asserts   : interpret asserts as contracts

.. Help ends: crosshair check --help

``watch``
=========

.. code-block::

    crosshair watch --help

.. Help starts: crosshair watch --help
.. code-block:: text

    usage: crosshair watch [-h] [--verbose] [--per_path_timeout FLOAT]
                           [--per_condition_timeout FLOAT] [--analysis_kind KIND]
                           FILE [FILE ...]

    The watch command continuously looks for contract counterexamples.
    Type Ctrl-C to stop this command.

    positional arguments:
      FILE                  File or directory to watch. Directories will be recursively analyzed.

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --per_path_timeout FLOAT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout FLOAT
                            Maximum seconds to spend checking execution paths for one condition
      --analysis_kind KIND  Kind of contract to check. By default, all kinds are checked.
                            See https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html
                                PEP316    : docstring-based contracts
                                icontract : decorator-based contracts
                                asserts   : interpret asserts as contracts

.. Help ends: crosshair watch --help

``diffbehavior``
================

.. code-block::

    crosshair diffbehavior --help

.. Help starts: crosshair diffbehavior --help
.. code-block:: text

    usage: crosshair diffbehavior [-h] [--verbose] [--per_path_timeout FLOAT]
                                  [--per_condition_timeout FLOAT]
                                  FUNCTION1 FUNCTION2

    Find differences in the behavior of two functions.
    See https://crosshair.readthedocs.io/en/latest/diff_behavior.html

    positional arguments:
      FUNCTION1             first fully-qualified function to compare (e.g. "mymodule.myfunc")
      FUNCTION2             second fully-qualified function to compare

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v         Output additional debugging information on stderr
      --per_path_timeout FLOAT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout FLOAT
                            Maximum seconds to spend checking execution paths for one condition

.. Help ends: crosshair diffbehavior --help
