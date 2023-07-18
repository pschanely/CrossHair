from io import StringIO

import pytest

from crosshair.core import deep_realize
from crosshair.core_and_libs import standalone_statespace
from crosshair.libimpl.iolib import BackedStringIO


@pytest.mark.parametrize("nl", [None, "", "\n", "\r\n", "\r"])
def test_StringIO_newlines(nl) -> None:
    text = "CR\rCRNL\r\nNL\nEND"
    concrete_input = StringIO(text, newline=nl)
    concrete_output = concrete_input.readlines()
    with standalone_statespace as space:
        symbolic_input = BackedStringIO(text, newline=nl)
        symbolic_output = deep_realize(symbolic_input.readlines())
    print(nl, concrete_output)
    assert symbolic_output == concrete_output
