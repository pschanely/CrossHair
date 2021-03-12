****************
IDE Integrations
****************

Install a plugin for your IDE to make it easier for CrossHair to help you:

* `VS Code`_
* `Emacs (flycheck)`_

.. _VS Code: https://marketplace.visualstudio.com/items?itemName=mristin.crosshair-vscode
.. _Emacs (flycheck): https://github.com/pschanely/emacs-flycheck-crosshair

If you make a plugin for your favorite editor (please do!),
submit a pull request to add it above.

The ``crosshair check [FILENAME]`` command will yield results in the same format
as the mypy type checker.
Namely, it exits with a non-zero code for errors, and produces lines formatted as
``{FILENAME}:{LINE_NUMBER}: error: {MESSAGE}``.