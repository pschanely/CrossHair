****************
IDE Integrations
****************

Consider installing a CrossHair plugin for your IDE:

* `Emacs`_
* `PyCharm`_
* `Thonny`_
* `VS Code`_

.. _Emacs: https://github.com/pschanely/emacs-flycheck-crosshair
.. _PyCharm: https://plugins.jetbrains.com/plugin/16266-crosshair-pycharm
.. _Thonny: https://pypi.org/project/thonny-crosshair/
.. _VS Code: https://marketplace.visualstudio.com/items?itemName=mristin.crosshair-vscode

Not using one of these? It's likely easy to integrate CrossHair with your IDE.
You can run crosshair using the Language Server Protocol (preferred), or parse its
command line output - the two options are explained below.
Finally, if you do publish a plugin for your favorite editor,
submit a pull request adding it to the list above!

Integrate using the Language Server Protocol
--------------------------------------------

Most IDEs support the
`Language Server Protocol (LSP) <https://microsoft.github.io/language-server-protocol/>`__,
either natively or through an extension/library.
It's best to run CrossHair via LSP: having a persistent process in the background
gives CrossHair a much better chance of finding counterexamples.
And, to make things simple, an LSP server already available when you install CrossHair.

You'll need to configure a command line so that your IDE can start the LSP server.
Set something like this: ``{PATH-TO-PROJECT-PYTHON}python -m crosshair server``.
After your IDE starts the server,
it will send notifications when files are opened or saved;
the server will continuously check them and send diagnostics (counterexamples) back.
One wrinkle: because CrossHair needs to run in the same Python environment as
the project, it's important to start the LSP server using the same python executable
as the project. You might also want check whether crosshair is installed in the
project's python environment before attempting to start the server.

Integrate by parsing command output
-----------------------------------

An alternative to using the Language Server Protocol is to have your IDE run a command
whenever a file is saved, and parse the results.
The ``crosshair check [FILENAME]`` command will yield results in the same format
as the mypy type checker.
Namely, it exits with a non-zero code for errors, and produces lines formatted as
``{FILENAME}:{LINE_NUMBER}: error: {MESSAGE}``.
You might even be able to copy an existing mypy plugin for your IDE.
