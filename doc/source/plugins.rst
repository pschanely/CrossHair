*******
Plugins
*******

Most commonly, plugin modules help CrossHair understand 3rd party packages that are
implemented in C. (also known as "extension modules")

CrossHair will generally understand 3rd party packages written in pure Python
out-of-the-box.

Using Plugins
=============

Just install the plugin package the same way you added "crosshair-tool".

If you write your own plugin, let us know so we can reference it here!

.. note::

    You can also register a plugin on a per-execution basis by using the
    ``--extra_plugin`` command line option, followed by your plugin files.

Writing Plugins
===============

**Re-writing classes and functions**

Typically, your plugin will re-implement the classes and functions of the native
package, and then tell CrossHair to use those instead.

Here is an example plugin package that attempts to mimic a hypothetical native Python
package called ``bunnies``.

.. code-block::

    # crosshair_plugin_bunnies.py

    from crosshair import register_patch, register_type, SymbolicFactory

    # import the original, native implementations:
    from bunnies import Bunny, introduce_bunnies

    # Replicate the native "Bunny" class in pure Python:

    class _Bunny:
        happiness: int
        def __init__(self, factory: SymbolicFactory):
            # CrossHair will pass your constructor a factory that you can use to create
            # more symbolic values of any other type.
            self.happiness = factory(int)

        def pet(self: _Bunny) -> None:
            self.happiness += 1

    # Replicate functions too:

    AnyBunny = Union[Bunny, _Bunny]  # arguments can be any kind of Bunny
    def _introduce_bunnies(bunny1: AnyBunny, bunny2: AnyBunny) -> None:
        bunny1.happiness += 1
        bunny2.happiness += 1
    

    # Tell CrossHair to use these implementations instead of the native ones:
    register_type(bunnies.Bunny, _Bunny)
    register_patch(bunnies.introduce_bunnies, _introduce_bunnies)


To let CrossHair know that your package is a plugin, define an "entry point" for your
distribution with a key of ``crosshair.plugin``, like so:

.. code-block::

    # setup.py

    from setuptools import setup
    setup(
        name="crosshair-plugin-bunnies",
        py_modules=["crosshair_plugin_bunnies"],
        entry_points={"crosshair.plugin": ["bunnies = crosshair_plugin_bunnies"]},
    )

**Adding contracts to external functions**

Re-writing classes and functions certainly takes some time, so here is another class of
plugins to specify contracts for external functions without re-implementing them. During
analysis, CrossHair will skip executing such functions. Instead, it will check if the
preconditions hold and then assume the postconditions are satisfied.

In addition to registering functions implemented in C, one might also wish to register
other Python functions, which are not fully checked by CrossHair (if you are using the
``--report_all`` option, for example).

As an example, suppose you want to use ``random.randint`` in your code but CrossHair
cannot exhaustively verify it because of its non-deterministic behavior.
You can register a contract for this function as follows:

.. code-block::

    from crosshair.register_contract import register_contract
    from random import Random
    register_contract(
        Random.randint,
        pre=lambda a, b: a <= b,
        post=lambda __return__, a, b: a <= __return__ and __return__ <= b,
    )

.. note::

    The names ``a`` and ``b`` above come from the definition of ``randint``.
    These names must be correct and CrossHair will throw an error if you register
    contracts with other argument names. In order to find the correct names, look for
    the source code of the function.


In case CrossHair did not find a signature for the function you are registering, it will
display a warning and might encounter an error while checking your code. This may happen
because CrossHair cannot infer the return type of the function. It will assume the
return type to be ``object``, which might be wrong. If you encounter such a warning, you
should register the signature for the funcion as well. As an example here, we will
register the function ``numpy.random.randint`` (note that numpy is a C module):

.. code-block::

    from crosshair.register_contract import register_contract
    import numpy as np
    from inspect import Parameter, Signature
    randint_sig = Signature(
        parameters=[
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("low", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter("high", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        ],
        return_annotation=int,
    )
    register_contract(
        np.random.RandomState.randint,
        pre=lambda low, high: low < high,
        post=lambda __return__, low, high: low <= __return__ and __return__ < high,
        sig=randint_sig,
    )

Now assume you write the following test:

.. code-block::

    import numpy as np
    def myrandom(a: int) -> int:
        """
        pre: a < 10
        post: _ > a
        """
        return np.random.randint(a, 10)


When you run ``crosshair check`` on this test file, with the above plugin, you will see
the fault:

.. code-block::

    error: false when calling myrandom(0)
    with crosshair.patch_to_return({<method 'randint' of 'numpy.random.mtrand.RandomState' objects>: [0]})

This is telling you that if you call ``myrandom(0)`` and ``randint`` returns ``0``, the
postcondition fails. Indeed, the postcondition is wrong and should be ``_ >= a``
instead!

.. note::

    The ``crosshair.patch_to_return(...)`` expression above may be used in a
    `with statement <https://docs.python.org/3/reference/datamodel.html#context-managers>`__
    to reproduce the failure.

.. note::

    You might have noticed that we registered ``np.random.RandomState.randint`` and not
    ``np.random.randint``. This is because the latter is a
    `bound function <https://www.pythontutorial.net/python-oop/python-methods/>`__
    (it is the method of a particular ``RandomState`` instance). However, we want to
    register the class function directly, so that our contract holds when calling
    ``randint`` on any ``RandomState`` instance. Note that for most functions, you
    should not encounter this problem, as bound functions are not common.
    For curious people: If you look into the source code of ``numpy.random.mtrand.pyx``,
    you will see how the bound function is defined: ``_rand = RandomState()`` and then
    ``randint = _rand.randint``. We indeed see that this is the method of a specific
    instance of ``RandomState``.
