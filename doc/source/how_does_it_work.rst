*****************
How Does It Work?
*****************

This is a small introduction to CrossHair's implementation.

Background
==========

We recommend the following reading, depending on your familiarity with other work in
this space.

* First, we can't recommend `<https://www.fuzzingbook.org/>`_ enough.
  CrossHair's approach largely corresponds to the chapter on
  `concolic fuzzing <https://www.fuzzingbook.org/html/ConcolicFuzzer.html>`_.
  Some parts of the
  `symbolic fuzzing <https://www.fuzzingbook.org/html/SymbolicFuzzer.html>`_
  chapter are relevant as well.
  Unlike concolic execution, CrossHair's values start each path execution as
  purely symbolic values, and do not get concrete values until they are needed.

* CrossHair uses the Z3 SMT solver to perform its deductions.
  `This Python-specific introduction <https://www.cs.tau.ac.il/~msagiv/courses/asv/z3py/guide-examples.htm>`_
  is a good place to get a feel for what it does.
  Please note that we'll say "Z3" below, but many statements about Z3 apply to all SMT
  solvers and/or the SMT-LIB language.

The Basic Idea
==============

CrossHair repeatedly calls the function you want to analyze and passes in special
objects that behave like things your function expects.

CrossHair doesn't do any kind of AST or bytecode analysis. It just calls your function.

These "special objects" behave differently on each execution.
That's how CrossHair explores different paths through your function.

These special CrossHair objects hold one or more Z3 expressions which are used in the Z3
solver.

When your function takes a parameter with some python type (say, ``int``) CrossHair will
supply an object of a custom CrossHair type (``SymbolicInt``), which holds a Z3
expression having some Z3 sort (``IntSort()``). Here are some examples:

+-------------+------------------+------------------------------------------------------+
| Python Type | CrossHair Type   | Z3 Sort                                              |
+=============+==================+======================================================+
| ``int``     | ``SymbolicInt``  | ``IntSort()``                                        |
+-------------+------------------+------------------------------------------------------+
| ``bool``    | ``SymbolicBool`` | ``BoolSort()``                                       |
+-------------+------------------+------------------------------------------------------+
| ``str``     | ``SymbolicStr``  | ``StringSort()``                                     |
+-------------+------------------+------------------------------------------------------+
| ``dict``    | ``SymbolicDict`` | ``ArraySort(K, V)`` and ``IntSort()`` for the length |
+-------------+------------------+------------------------------------------------------+

Let's Explore
=============

We can initialize a CrossHair object by giving it a name::

    >>> from crosshair.libimpl.builtinslib import SymbolicInt

    >>> crosshair_x = SymbolicInt('x')

We can access the ``.var`` attribute of any CrossHair object to get
the Z3 variable(s) that it holds:

    >>> crosshair_x.var
    x
    >>> type(crosshair_x.var)
    <class 'Z3.Z3.ArithRef'>


This takes the Z3 variable we just defined and adds one to it::

    >>> import Z3

    >>> expr = crosshair_x.var + Z3.IntVal(1)
    >>> expr
    x + 1
    >>> type(expr)
    <class 'Z3.Z3.ExprRef'>

We can create CrossHair objects not only for fresh variables, but
also for Z3 expressions.
So, if we wanted to wrap ``x + 1`` back into a CrossHair object,
we'd write::

    >>> SymbolicInt(crosshair_x.var + Z3.IntVal(1))

The ``SymbolicInt`` class defines the ``__add__`` method so that you don't
have to spell that out, though. You can just say ``crosshair_x + 1``, and
``SymbolicInt`` does the necessary unwrapping and re-wrapping::

    >>> type(crosshair_x + 1)
    <class 'crosshair.libimpl.builtinslib.SymbolicInt'>

``SymbolicInt`` also defines the comparison methods so that they return symbolic
booleans::

    >>> type(crosshair_x >= 0)
    <class 'crosshair.libimpl.builtinslib.SymbolicBool'>

The symbolic boolean holds an equivalent Z3 expression::

    >>> (crosshair_x >= 0).var
    0 <= x


So far, everything is symbolic. But eventually, the Python interpreter
needs a real value; consider::

    if crosshair_x > 0:
        print('bigger than zero')

Should this execute the print or not? When python executes the ``if``
statement, it calls ``__bool__`` on the ``SymbolicBool`` object. This method
does something very special. It consults Z3:

* If the Z3 boolean expression must be True (or False), just return
  that value.

* Otherwise, decide it to be True or False randomly. Take that decision
  and add it to the set of Z3 constraints for this execution path.
  Return the (concrete) bool that we decided.

CrossHair will remember what decisions it has made so that
it can make different decisions on future executions. Ultimately,
we're looking for some target thing to happen: an exception to be
raised, or a postcondition to return False. When that happens,
we ask Z3 for a model and report it as a counterexample.

That's the core of how CrossHair works.


Devil in the Details
====================

Simple right?

Well, if there is an accomplishment about CrossHair, it's that it
tries hard to get the details right. And there are **a lot** of
details.

Here are some of the topics that aren't yet discussed. Reach out to help us prioritize
documenting them!


* Balancing the amount of work done inside and outside the solver.
* Developing heuristics for effective path exploration.
* Dealing with the cases that Z3 cannot. (concrete/symbolic scaling)
* Interpreting logic that's implemented in C.
* Reconciling semantic differences between Python and Z3.
* Dealing with mutable values.
* Dealing with potentially aliased mutable values (x is y).
* Creating symbolics for your custom classes.
* Reconciling error behavior (ValueErrors, TypeErrors).
* Implicitly converting types accurately.
* Managing evaluation order. (under-approximation and over-approximation tactics)
* Creating symbolics for base classes, or even for ``object``.
