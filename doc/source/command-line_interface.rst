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
        check               Analyze a file
        watch               Continuously watch and analyze a directory
        diffbehavior        Find differences in behavior between two functions

    optional arguments:
      -h, --help            show this help message and exit

.. Help ends: crosshair --help

``check``
=========

.. code-block::

    crosshair check --help

.. Help starts: crosshair check --help
.. code-block:: text

    usage: crosshair check [-h] [--verbose] [--per_path_timeout PER_PATH_TIMEOUT]
                           [--per_condition_timeout PER_CONDITION_TIMEOUT]
                           [--report_all] [--analysis_kind ANALYSIS_KIND]
                           F [F ...]

    positional arguments:
      F                     file or fully qualified module, class, or function

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v
      --per_path_timeout PER_PATH_TIMEOUT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout PER_CONDITION_TIMEOUT
                            Maximum seconds to spend checking execution paths for
                            one condition
      --report_all          Output analysis results for all postconditions (not
                            just failing ones)
      --analysis_kind ANALYSIS_KIND
                            Kinds of analysis to perform.

.. Help ends: crosshair check --help

``watch``
=========

.. code-block::

    crosshair watch --help

.. Help starts: crosshair watch --help
.. code-block:: text

    usage: crosshair watch [-h] [--verbose] [--per_path_timeout PER_PATH_TIMEOUT]
                           [--per_condition_timeout PER_CONDITION_TIMEOUT]
                           [--analysis_kind ANALYSIS_KIND]
                           F [F ...]

    positional arguments:
      F                     file or directory to analyze

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v
      --per_path_timeout PER_PATH_TIMEOUT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout PER_CONDITION_TIMEOUT
                            Maximum seconds to spend checking execution paths for
                            one condition
      --analysis_kind ANALYSIS_KIND
                            Kinds of analysis to perform.

.. Help ends: crosshair watch --help

``diffbehavior``
================

.. code-block::

    crosshair diffbehavior --help

.. Help starts: crosshair diffbehavior --help
.. code-block:: text

    usage: crosshair diffbehavior [-h] [--verbose]
                                  [--per_path_timeout PER_PATH_TIMEOUT]
                                  [--per_condition_timeout PER_CONDITION_TIMEOUT]
                                  fn1 fn2

    positional arguments:
      fn1                   first module+function to compare (e.g.
                            "mymodule.myfunc")
      fn2                   second function to compare

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v
      --per_path_timeout PER_PATH_TIMEOUT
                            Maximum seconds to spend checking one execution path
      --per_condition_timeout PER_CONDITION_TIMEOUT
                            Maximum seconds to spend checking execution paths for
                            one condition

.. Help ends: crosshair diffbehavior --help
