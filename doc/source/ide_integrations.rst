****************
IDE Integrations
****************
Optimally, CrossHair wants to run in the background so it can have plenty of
time to think.
However, IDE integrations can still be used to catch easy cases.

* `Emacs (flycheck) <https://github.com/pschanely/emacs-flycheck-crosshair>`_

If you make a plugin for your favorite editor (please do!),
submit a pull request to add it above.

The ``crosshair check [FILENAME]`` command will yield results in the same format
as the mypy type checker.
Namely, it gives a non-zero exit for errors, and lines formatted as
``{FILENAME}:{LINE_NUMBER}:error:{MESSAGE}``.