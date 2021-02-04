import icontract


@icontract.require(lambda x: x > 0)
@icontract.ensure(lambda result: result > 0)
def some_func(x: int) -> int:
    # The constant makes the result negative.
    return x - 1000
