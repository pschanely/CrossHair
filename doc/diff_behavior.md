## The `diffbehavior` command

Are these two functions equivalent?
```py
# foo.py

def cut1(a: List[int], i: int) -> None:
  a[i:i+1] = []

def cut2(a: List[int], i: int) -> None:
  a[:] = a[:i] + a[i+1:]
```

Almost! But not quite.

CrossHair's `diffbehavior` command can help you find out:

```
$ crosshair diffbehavior foo.cut1 foo.cut2

Given: (a=[9, 0], i=-1),
  foo.cut1 : after execution a=[9, 0]
  foo.cut2 : after execution a=[9, 9, 0]
```

### How do I try it?

```
$ pip install crosshair-tool
$ crosshair diffbehavior <module>.<function> <module>.<function>
```

### `diffbehavior` your own code changes

Use `git worktree` to create a quick, unmodified source tree, and then use
`crosshair diffbehavior` to compare your changes to head.

```
# Let's say we edit the clean() function in foo.py

# Step 1: Create an unmodified source tree under a directory named "clean":
$ git worktree add --detach clean

# Step 2: Have CrossHair try to detect a difference:
$ crosshair diffbehavior foo.cut clean.foo.cut

# Step 3: Remove the "clean" directory when you're done:
$ git worktree remove clean
```

### An example shell function
If you find yourself doing this often, make yourself a function or script.
For example, this function in your `~/.bashrc` file:
```
diffbehavior() {
    git worktree add --detach _clean || exit 1
    crosshair diffbehavior "$1" "_clean.$@"
    git worktree remove _clean
}
```
will let you diff your uncommitted changes very easily:
```
$ diffbehavior foo.cut
...
```

### Refactoring? Use `diffbehavior` to make sure it's correct.

Say we start with this:
```py
def longest_str(items: List[str]) -> str:
  longest = ''
  for item in items:
    if len(item) > len(longest):
      longest = item
  return longest
```
... and change it to this:
```py
def longest_str(items: List[str]) -> str:
  return max(items,
             key=lambda item: len(item),
             default='')
```
We can use [the shell function above](#an-example-shell-function) to help
make sure the code doesn't work differently:
```
$ diffbehavior longest_str
No differences found. (attempted 15 iterations)
Consider trying longer with: --per_condition_timeout=<seconds>
```


### Making real changes? `diffbehavior` helps write your new unit tests.

Say we start with this:
```py
def isack(s: str) -> bool:
    if s in ('y', 'yes'):
        return True
    return False
```
... and change it to this:
```py
def isack(s: str) -> bool:
    if s in ('y', 'yes', 'Y', 'YES'):
        return True
    if s in ('n', 'no', 'N', 'NO'):
        return False
    raise ValueError('invalid ack')
```
We can use [the shell function above](#an-example-shell-function) to find
useful inputs for testing:
```
$ diffbehavior foo.isack
Given: (s='\x00'),
         foo.isack : returns False
  _clean.foo.isack : raises ValueError('invalid ack')
Given: (s='YES'),
         foo.isack : returns False
  _clean.foo.isack : returns True
```
CrossHair reports examples in order of added coverage, descending, so consider
writing your unit tests using such inputs, from the top-down.

But don't do it blindly! CrossHair doesn't always give plesant examples;
instead of using `'\x00'`, you should just use `'a'` to cover the same logic.

## Caveats

* This feature, as well as CrossHair generally, is a work in progress. If you
  are willing to try it out, thank you! Please file bugs or start discussions to
  let us know how it went.
* Be aware that the absence of an example difference does not guarantee that the
  functions are equivalent.
* CrossHair likely won't be able to detect differences in complex code.
  * Target it at the smallest piece of logic possible.
* Your arguments have to be deep-copy-able and equality-comparable. (this is so
  that we can detect code that mutates them)
* Only deteministic behavior can be analyzed.
  (your code always does the same thing when starting with the same values)
* Be careful: CrossHair will actually run your code and may apply any arguments
  to it.
  * If you run CrossHair on code calling
    [shutil.rmtree](https://docs.python.org/3/library/shutil.html#shutil.rmtree),
    you **will** destroy your filesystem.
