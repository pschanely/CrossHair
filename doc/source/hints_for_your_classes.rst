
**********************
Hints for Your Classes
**********************

CrossHair can reason about classes that you write, and may likely do the right
thing out of the box.

To be sure it does, though, **add types for your __init__() arguments**.
When needed, CrossHair will attempt to construct your class with symbolic arguments.
To do this, it looks up types on the parameters  to the ``__init__`` method.

If you use `Python's dataclass module`_, your generated ``__init__`` arguments will be
already typed.
That means CrossHair will also work out-of-the-box with definitions like this::

  import dataclasses
  @dataclasses.dataclass
  class Person:
    name: str
    age: int

Customizing Creation
====================


There are a variety of times where you may need to customize the way CrossHair
constructs instances of your class.
For example,

* You are using some sort of class wrapper that hides the types on the constructor
  arguments.
* You don't have control over the class that you want CrossHair to reason about.
* The class has valid/reachable states that aren't directly constructable.
* The type is implemented in C.

.. note::
    This capabliity and the interface to it is under active development and unstable.
    That said, if you are willing to try it out, please ask questions and let us know
    how it goes.

In such cases, you may register a creation callback with
:func:`crosshair.register_type`.
Your creation callback will be given a :class:`crosshair.SymbolicFactory` instance that
can create other kinds of symbolic objects, which you should use to initialize your
instance.


Example
-------

Suppose you had a Counter class that could have any count, but is only construct-able
at zero::

  class Counter:
    def __init__(self):
      self.count = 0
    # ... more methods ...

By default, CrossHair would assume that the count must always be zero, even though
methods or direct assignments could change it.
To ensure CrossHair knows that any integer count is possible, we will tell
CrossHair to use a custom function when creating ``Counter`` instances, like so::

  import crosshair
  def symbolc_counter(factory: crosshair.SymbolicFactory) -> Counter:
    counter = Counter()
    counter.count = factory(int)  # count is now a symbolic holding any integer value
    return counter
  crosshair.register_type(Counter, symbolic_counter)

Note that we might want to further ensure that the count is greater than zero.
The best way to do this is to add an invariant to the class. (e.g. "self.count >= 0")
CrossHair will make sure all invariants hold for symbolic instances after your creation
function produces them.


.. _Python's dataclass module: https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass
