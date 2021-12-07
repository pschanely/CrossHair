************
Introduction
************

Crosshair is an analysis tool for Python that blurs the line between testing and
type systems.

If you have a function with `type annotations`_ and add a
contract :ref:`in a supported kind of contracts <Kinds of Contracts>`,
CrossHair will attempt to find counterexamples for you:

.. _type annotations: https://www.python.org/dev/peps/pep-0484/

.. image:: duplicate_list.gif
    :width: 610
    :height: 192
    :alt: Animated GIF demonstrating the verification of a python function

CrossHair works by repeatedly calling your functions with symbolic inputs.
It uses an `SMT solver`_ (a kind of theorem prover) to explore viable execution
paths and find counterexamples for you.
This is not a new idea; an approach for Python was first described in
`this paper`_.
However, to my knowledge, CrossHair is the most complete implementation of the
idea: it supports symbolic lists, dictionaries, sets, and
custom mutable objects.

.. _SMT solver: https://en.wikipedia.org/wiki/Satisfiability_modulo_theories
.. _this paper: https://hoheinzollern.files.wordpress.com/2008/04/seer1.pdf

Try CrossHair right now, in your browser, at `crosshair-web.org`_!

.. _crosshair-web.org: https://crosshair-web.org
