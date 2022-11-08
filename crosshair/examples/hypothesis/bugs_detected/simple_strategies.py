import string

from hypothesis import given
from hypothesis import strategies as st


@given(st.integers())
def positive(s):
    assert s >= 0


@given(st.tuples(st.booleans(), st.booleans()))
def tuples(t):
    assert t != (False, True)


@given(st.text(alphabet=string.ascii_uppercase, min_size=3, max_size=3))
def not_us_currency(s):
    assert s != "USD"


@given(st.characters(whitelist_categories=["Nd"]))
def number_chars(c):
    assert c in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


# TODO: Various example cases to work on below

# @given(
#     st.integers(min_value=100), st.one_of(st.just(2), st.just(1)), st.integers(-1, 1)
# )
# def big_sum(a, b, c):
#     assert a + b + c > 100


# @given(st.text())
# def not_u_string(s):
#     assert s != "U"


# @given(value=st.floats(allow_nan=True))
# def equal_to_self(f):
#     assert f == f

# @given(st.dictionaries(st.text(), st.text()))
# def challenge(d):
#     assert d.get("a") != "b"


# @given(st.bits(nbits))
# @given(st.binary())
# @given(st.booleans())
# @given(st.floats(min_value=-0.05, max_value=0.05))

# Keys = typing.Set[typing.Hashable]
# @given(keys=infer)
# def test_infer(keys: Keys) -> None:
#  pass

# @given(
#     n_periods1=st.integers(1, 10),
#     n_periods2=st.integers(1, 10),
#     log_base_diff=arrays(
#         np.float64,
#         st.integers(2, 10),
#         elements=st.floats(-2.0, 2.0, width=64),
#     ),
# )

# @hypothesis.given(
#     arr=arrays(dtype=np.float64,
#                shape=st.integers(min_value=3, max_value=1000),
#                elements=st.floats(allow_infinity=False, allow_nan=False,
#                                   min_value=-1e300, max_value=1e300)))
