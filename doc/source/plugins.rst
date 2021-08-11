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

Writing Plugins
===============

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
