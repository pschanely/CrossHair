from hypothesis import given, infer
from hypothesis import strategies as st


@given(st.integers())
def positive(s):
    assert s >= 0
