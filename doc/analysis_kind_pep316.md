## PEP 316 Contracts

[PEP 316](https://www.python.org/dev/peps/pep-0316/) is an abandoned PEP
for design-by-contract in Python. CrossHair can check such contracts with
the `pep316` analysis kind:

```shell
$ crosshair [check|watch] --analysis_kind=PEP316 <filename>
```


PEP316 pairs well with
[doctest](https://docs.python.org/3/library/doctest.html).
Doctest is great for illustrative examples and CrossHair can document behavior
more holistically. Some kinds of projects may be able to skip unittest/pytest
entirely.

![Image showing a comment block with doctest and CrossHair conditions](even_fibb.png)

## How to Write Contracts

See the [PEP 316](https://www.python.org/dev/peps/pep-0316/) specification for details. In short:
- Place contracts inside the docstrings for functions.
- Declare your post-conditions (what you expect to be true of the function's return value) like this: <BR>`post: __return__ > 0`
  - If you like, you can use a single underscore (`_`) as a short-hand for `__return__`.
- Functions are checked if they have at least one post-condition line in their docstring.
- Declare your pre-conditions (what you expect to be true of the function's inputs) like this: <BR>`pre: x < y`
- Declare that your function mutates arguments with square brackets after the `post` keyword.
  - When doing so, the old values of the arguments are available in a special object called `__old__`: <BR>`post[x]: x > __old__.x`
  - Comparison for the purposes of mutation checking is a "deep" comparison.
  - Use empty square brackets to assert that the function does not mutate any argument.
- If your function can validly raise certain exceptions, declare them like this: <BR>`raises: IndexError, ZeroDivisionError`
- Declare class invariants in the class's docstring like this: <BR>`inv: self.foo < self.bar`
  - Class invariants apply additional pre- and post-conditions to each member function.
- Note: Unlike contracts on standalone functions, contracts on class methods often encourage/require contracts on the entire class.
  - This is because you usually need invariants on the class to describe what states are valid, and then every method must
    be shown to preserve those invariants.

You can find examples in the [examples/](https://github.com/pschanely/CrossHair/tree/master/crosshair/examples) directory 
and try it in your browser at [crosshair-web.org](https://crosshair-web.org).



