************
Contributing
************

Coordinate First
================

Before you create a pull request, please `create a new issue`_ first
or check in on `gitter`_ to coordinate.

It might be that we are already working on the same or similar feature, but we
haven't made our work visible yet.

.. _create a new issue: https://github.com/pschanely/CrossHair/issues/new/choose
.. _gitter: https://gitter.im/Cross_Hair/Lobby

Create a Development Environment
================================

We usually develop in a `virtual environment`_.
To create one, change to the root directory of the repository and invoke:

.. code-block::

    python -m venv venv


You need to activate it. On *nix (Linux, Mac, *etc.*):

.. code-block::

    source venv/bin/activate

and on Windows:

.. code-block::

    venv\Scripts\activate

.. _virtual environment: https://docs.python.org/3/tutorial/venv.html

Install Development Dependencies
================================

Once you activated the virtual environment, you can install the development
dependencies using ``pip``:

.. code-block::

    pip3 install --editable .[dev]

The `--editable <pip-editable_>`_ option is necessary so that all the changes
made to the repository are automatically reflected in the virtual environment
(see also `this StackOverflow question <pip-editable-stackoverflow_>`_).

.. _pip-editable: https://pip.pypa.io/en/stable/reference/pip_install/#install-editable
.. _pip-editable-stackoverflow: https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install

Pre-commit Checks
=================

We provide a battery of pre-commit checks to make the code uniform and
consistent across the code base.

We use `black`_ to format the code and use the default maximum line length of
88 characters.

.. _black: https://pypi.org/project/black/

To run all pre-commit checks, run from the root directory:

.. code-block::

    pre-commit run --all-files

The pre-commit hooks also run as part of our continuous integration pipeline.

Write Commit Message
====================

We follow Chris Beams' `guidelines on commit messages`_:

1) Separate subject from body with a blank line
2) Limit the subject line to 50 characters
3) Capitalize the subject line
4) Do not end the subject line with a period
5) Use the imperative mood in the subject line
6) Wrap the body at 72 characters
7) Use the body to explain *what* and *why* vs. *how*

.. _guidelines on commit messages: https://chris.beams.io/posts/git-commit/


Contributor List
================

In order of inital commit. Many thanks!

* `Phil Schanely <https://twitter.com/pschanely>`_
* `Edward Haigh <https://github.com/oneEdoubleD>`_
* `Saul Shanabrook <https://github.com/saulshanabrook/>`_
* `Marko Ristin <https://github.com/mristin>`_
* `Rik de Kort <https://github.com/Rik-de-Kort>`_
* `Dmytro Khomutnyk <https://github.com/XoMute>`_
* `Martin Matějek <https://github.com/mmtj>`_
* `Loïc Montandon <https://github.com/lmontand>`_
* `Alec Delaney <https://github.com/tekktrik>`_
