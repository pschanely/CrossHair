from enum import IntEnum

from crosshair.z3util import z3IntVal


class IntSubClass(IntEnum):
    FIRST = 1


def test_intval_on_int_enum():
    z3IntVal(IntSubClass.FIRST)
