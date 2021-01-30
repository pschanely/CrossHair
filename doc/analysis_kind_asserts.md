
## Assert-based Contracts

This is the lowest-investment way to use contracts with CrossHair. You just use
regular
[assert statements](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)
in your code. There's **no library to import** and **no syntax to learn**: just
use assert statements.

To try it out, use the "asserts" analysis kind:
```
$ crosshair [check|watch] --analysis_kind=asserts <filename>
```

### How It Works

CrossHair will analyze any function that starts with one or more assert
statements. (it will ignore any function that does not!)

```py
# foo.py
from typing import List

def remove_smallest(numbers: List[int]) -> None:
  ''' Removes the smallest number in the given list. '''
 
  # The precondition: CrossHair will assume this to be true:
  assert len(numbers) > 0

  smallest = min(numbers)

  numbers.remove(smallest)

  # The postcondition: CrossHair will find examples to make this be false:
  assert len(numbers) == 0 or min(numbers) > smallest
```

The leading assert statement(s) are considered to be preconditions: CrossHair
will try to find inputs that make these true.

After the precondition asserts, we expect the remainder of the function to behave
safely. Namely,
* it will not fail on any later assert.
* it will not raise any exception.

The example postcondition above isn't quite correct: it fails when there are duplicates
of the smallest number. CrossHair can tell you this:

```
$ crosshair check --analysis_kind=asserts foo.py
foo.py:14:error:AssertionError:  when calling remove_smallest(numbers = [0, -1, -177, -178, -178])
```
