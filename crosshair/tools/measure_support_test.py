"""Tests for the demo-readability (``_noise``/``_pair_noise``), echo-detection
(``_is_echo``), and echo-witness-upgrade (``_upgrade_echo_witness``) helpers that
shape the support map's generated demo links."""

from crosshair.tools.measure_support import (
    _is_echo,
    _noise,
    _pair_noise,
    _upgrade_echo_witness,
)


def test_noise_ranks_plain_ascii_below_escapes_below_astral():
    plain = _noise("abcde")
    escapes = _noise("\x00\x01\x02")
    astral = _noise("𰄜i\x13ât")
    assert plain < escapes < astral


def test_noise_prefers_shorter_when_both_plain():
    assert _noise("ab") < _noise("abcdef")


def test_noise_handles_non_string_values():
    # ints / containers render via repr; small + ASCII scores low
    assert _noise(5) < _noise(10**40)
    assert _noise([0, 0, 0]) < _noise(["𰄜", "𝤬"])


def test_pair_noise_sums_inputs_and_output():
    readable = (("abc",), "abc")
    noisy = (("𰄜i\x13ât",), "𰄜i")
    assert _pair_noise(readable) < _pair_noise(noisy)


def test_is_echo_true_when_output_equals_that_argument():
    # list.copy: output equals the receiver (argument 0)
    assert _is_echo(([1, 2, 3],), [1, 2, 3], 0) is True


def test_is_echo_false_when_output_differs():
    # a + a doubles the list -- not identity in the receiver
    assert _is_echo(([1, 2, 3],), [1, 2, 3, 1, 2, 3], 0) is False


def test_is_echo_targets_the_named_argument_only():
    # output equals argument 1, not argument 0
    assert _is_echo((7, 3), 3, 0) is False
    assert _is_echo((7, 3), 3, 1) is True


def test_is_echo_swallows_uncomparable_values():
    class _Boom:
        def __eq__(self, other):
            raise RuntimeError("no compare")

    assert _is_echo((_Boom(),), _Boom(), 0) is False


def test_upgrade_echo_witness_gives_up_on_identity():
    # A genuinely identity-in-that-arg op yields no non-echo sample, so the demo-only
    # upgrade returns None (the cell keeps its correctly-flagged echo demo) WITHOUT
    # ever reaching the expensive CrossHair inversion.
    params = [("a", "str", str)]
    identity = lambda a: a  # noqa: E731  -- every sample echoes
    assert (
        _upgrade_echo_witness(
            "", params, "a", identity, 0, None, 3, "builtins", "x.identity"
        )
        is None
    )
