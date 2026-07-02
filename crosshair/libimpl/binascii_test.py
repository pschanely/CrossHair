import binascii
import sys
from array import array

import pytest

from crosshair.behavior_compare import summarize_execution
from crosshair.tracers import ResumedTracing


@pytest.mark.parametrize(
    "input_bytes",
    [
        b"",  # empty
        b"//==",  # padding 2
        b"AAA=",  # padding 1
        b"9999",  # no padding
        b"3=4=",  # discontinuous padding
        b"34=&=",  # unexpected chars in padding
        b"34=",  # wrong padding
        b"333====",  # over-padding
        b"This/12+yearOld/Fox=",  # valid, long
        "",  # empty string
        "9009",  # nonempty string
        "\u2165",  # unicode string
    ],
)
@pytest.mark.parametrize("strict_mode", [True, False])
def test_base64_decode(space, input_bytes, strict_mode):
    kw = {"strict_mode": strict_mode} if sys.version_info >= (3, 11) else {}
    concrete_result = summarize_execution(
        binascii.a2b_base64, (input_bytes,), kw, detach_path=False
    )
    with ResumedTracing():
        symbolic_result = summarize_execution(
            binascii.a2b_base64, (input_bytes,), kw, detach_path=False
        )
    assert concrete_result == symbolic_result


@pytest.mark.parametrize("newline", [True, False])
@pytest.mark.parametrize(
    "input_bytes",
    [
        b"",  # empty
        b"H",
        b"Ha",
        b"Hai",
        b"Hair",
        "",  # empty string
        "Hair",  # nonempty string
        "\u2165",  # unicode string
        bytearray(b"Hair"),  # bytearray
        memoryview(b"Hair"),  # memoryview
        array("B", b"Hair"),  # array
    ],
)
def test_base64_encode(space, input_bytes, newline):
    kw = {"newline": newline}
    concrete_result = summarize_execution(
        binascii.b2a_base64, (input_bytes,), kw, detach_path=False
    )
    with ResumedTracing():
        symbolic_result = summarize_execution(
            binascii.b2a_base64, (input_bytes,), kw, detach_path=False
        )
    assert concrete_result == symbolic_result


# b2a_base64 gained wrapcol/padded/alphabet keywords in 3.15 (base64.b64encode &
# friends now route through them), so power those paths.
@pytest.mark.skipif(
    sys.version_info < (3, 15), reason="wrapcol/padded/alphabet are new in 3.15"
)
@pytest.mark.parametrize("wrapcol", [0, 4, 5, 8, 76])
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize(
    "alphabet",
    [
        None,  # default alphabet
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",  # urlsafe
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@$",  # altchars
    ],
)
def test_base64_encode_315_kwargs(space, wrapcol, padded, alphabet):
    input_bytes = b"\xc2\xe3k" * 9 + b"\xff"  # spans several wrap lines, uneven tail
    kw = {"wrapcol": wrapcol, "padded": padded}
    if alphabet is not None:
        kw["alphabet"] = alphabet
    concrete_result = summarize_execution(
        binascii.b2a_base64, (input_bytes,), kw, detach_path=False
    )
    with ResumedTracing():
        symbolic_result = summarize_execution(
            binascii.b2a_base64, (input_bytes,), kw, detach_path=False
        )
    assert concrete_result == symbolic_result
