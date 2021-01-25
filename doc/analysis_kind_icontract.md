
## icontract checking

CrossHair supports checking [icontract](https://github.com/Parquery/icontract)
postconditions and invariants.

Use the "icontract" analysis kind:
```
$ crosshair [check|watch] --analysis_kind=icontract <filename>
```

### Things to know

* CrossHair will only analyze functions that have at least one postcondition
  (`@icontract.ensure`).
* CrossHair will actually invoke the analyzed code with arbitrary arguments -
  ensure you do not point it at code that uses the disk or network.
* See more caveats and background in CrossHair's [README](../README.md).
