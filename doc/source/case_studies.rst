############
Case Studies
############


Contractual SemVer
==================

`Contractual SemVer <https://github.com/pschanely/contractual-semver>`__
uses contracts to determine what changes are backwards compatible.

**For library authors**

Specify contracts for your public functions and classes.
These contract let library consumers know what they can expect from your library.

Minor and patch version increments may be accompanied by:

- Broader preconditions (you accept new inputs)
- Narrower postconditions (you're guaranteeing more about your output)

Other contract changes require a major version increment.

In addition, use a ``# crosshair: specs_complete=True`` directive comment in an
appropriate module or ``__init__.py``.
Then, if your consumers also use CrossHair, they'll see *any* possible behavior
meeting the contracts. (see below)

**For library consumers**

If the library that you're using applies the ``specs_complete`` directive described
above, CrossHair can help you.
When CrossHair runs, it will freely invent any behavior for the library, so long as that
behavior conforms to the contracts.

Even if you don't want to use contracts in *your* project, you can annotate your
existing unit tests with the trivial contract of ``"""post: True"""``, and CrossHair
will tell you when your unit tests have expectations exceeding the guarantees of the
library you're using.

**An Example**

Suppose the library author releases a function to get names for dogs.
This function might include preconditions and postconditions, describing what behaviors
the library consumer can depend upon.

    >>> from typing import List
    >>> def dog_names(limit: int) -> List[str]:
    ...     """
    ...     Do not give us a negative limit:
    ...     pre: limit >= 0
    ...
    ...     All dog names are non-empty:
    ...     post: all(len(name) > 0 for name in __return__)
    ...
    ...     We never return more names than the limit you provide:
    ...     post: len(__return__) <= limit
    ...
    ...     We will provide at least 3 names:
    ...     post: limit > 3 or len(__return__) == limit
    ...     """
    ...     # crosshair: specs_complete=True
    ...     return ["Toto", "Clifford", "Fido", "Doge", "Lassie"][:limit]

.. note::

    The library author may guarantee less than the implementation - in this example,
    5 dog names can be provided, but only 3 names are guaranteed by the contract.


And then, let's suppose the library consumer wants to display a table of locations and
dogs:

    >>> def display_in_locations(locations: List[str]):
    ...     for dog_name, location in zip(dog_names(len(locations)), locations):
    ...         yield dog_name.ljust(10) + "is in" + location.rjust(10)

What test should the consumer write? Here is a quick (but brittle) test:

    >>> def test_display_in_locations():
    ...     """ post: True """  #  <- The trivial contract enables CrossHair analysis
    ...     lines = display_in_locations(["Halifax", "New York"])
    ...     assert lines == [
    ...         "Clifford  is in   Halifax",
    ...         "Fido      is in  New York",
    ...     ]

This test passes, but it replies on specific dog names from the library.
It could easily break when the library is upgraded.

When the consumer runs ``crosshair watch`` on this test file, they will see the fault:

.. code-block::

    AssertionError when calling test_display_in_locations()
    with crosshair.patch_to_return({dog_names: [['a', 'a']]})

This is telling the library consumer that if ``dog_names`` returns ``['a', 'a']``, the
assertion fails. Indeed, it will fail for nearly any set of dog names!

.. note::

    The ``crosshair.patch_to_return(...)`` expression above may be used in a
    `with statement <https://docs.python.org/3/reference/datamodel.html#context-managers>`__
    to reproduce the failure.

The library consumer might improve this test by comparing the suffixes only:

    >>> def test_display_in_locations():
    ...     lines = display_in_locations(["Halifax", "New York"])
    ...     assert [l[10:] for l in lines] == [
    ...         "   Halifax",
    ...         "  New York",
    ...     ]

CrossHair detects one more problem here. If the dog name exceeds 10 characters, it will
spill over into the suffix. The consumer can avoid this by clipping all dog names to
10 characters. (and they should, since the library author has made no guarantees about
the maximum length of dog names)
