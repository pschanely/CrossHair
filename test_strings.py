from crosshair import *

def plural(s :isstring) -> isstring:
    return s + "s"

def _assert_plural_EndsWithS(s :isstring) -> istrue:
    return plural(s)[-1] == "s"

def _assert_plural_Cats() -> istrue:
    return plural("cat") == "cats"

def _assert_plural_Length(s :isstring) -> istrue:
    return len(plural(s)) == len(s) + 1

def _assert_double_plural_Length(s :isstring) -> istrue:
    return len(plural(plural(s))) >= 2

