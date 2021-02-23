**********************
What Code Is Analyzed?
**********************

Let's look at how you can target code for analysis, and what CrossHair
actually executes during analysis.

Targeting
=========

You can run the ``crosshair check`` command on:

* Directories. e.g. ``crosshair check mypkg/``
* Files. e.g. ``crosshair check mypkg/foo.py``
* File and line number. e.g. ``crosshair check mypkg/foo.py:23``
* Modules. e.g. ``crosshair check mypkg.foo``
* Classes. e.g. crosshair ``check mypkg.foo.MyClass``
* Functions or methods. e.g. crosshair ``check mypkg.foo.MyClass.my_method``

The ``crosshair watch`` command allows only file and directory arguments. (because it's
driven from file modify times)

CrossHair's analysis may be further restricted by special comments in your code, like
this::

    def foo():
        # crosshair: off
        pass

Directives may appear anywhere in the body of the function or method.

Directives may also appear at the top-level of a file, or in the ``__init__.py`` file
of a package.
You may also use a ``# crosshair: on`` comment to re-enable analysis as necessary.


What code is executed when CrossHair runs?
==========================================

CrossHair works by repeatedly calling the targeted functions with special values.

It may or may not execute your preconditions and postconditions.
It'll usually execute the code of subroutines as well, but doesn't always, and may
execute that logic out-of-order.
Mostly, you don't need to worry about these details, but some of these effects may
become visible if your code calls ``print()``, for instance.
(note further that printing symbolic values will force them to take on concrete values
and will hamper CrossHair's ability to effectively analyze your code!)

Because of the wide variety of things CrossHair might do, never target code that can
directly or indirectly call mutating side-effects.
CrossHair puts some protections in place (via ``sys.addaudithook``) to prevent disk
and network access, but this protection is not perfect. (notably, it will not
prevent actions taken by C-based modules)
