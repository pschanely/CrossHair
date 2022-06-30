import sys
import unicodedata

from crosshair.core import register_patch, with_realized_args
from crosshair.libimpl.builtinslib import AnySymbolicStr, SymbolicInt
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing
from crosshair.unicode_categories import UnicodeMaskCache


def _decimal(ch: str) -> float:
    codepoint = ord(ch)
    with NoTracing():
        if not isinstance(ch, AnySymbolicStr):
            return unicodedata.decimal(ch)
        space = context_statespace()
        smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
        cache = space.extra(UnicodeMaskCache)
        if not space.smt_fork(cache.decimal_exists()(smt_codepoint)):
            raise ValueError
        return SymbolicInt(cache.decimal_int()(smt_codepoint))


def _digit(ch: str) -> float:
    codepoint = ord(ch)
    with NoTracing():
        if not isinstance(ch, AnySymbolicStr):
            return unicodedata.digit(ch)
        space = context_statespace()
        smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
        cache = space.extra(UnicodeMaskCache)
        if not space.smt_fork(cache.digit_exists()(smt_codepoint)):
            raise ValueError
        return SymbolicInt(cache.digit_int()(smt_codepoint))


def _numeric(ch: str) -> float:
    codepoint = ord(ch)
    with NoTracing():
        if not isinstance(ch, AnySymbolicStr):
            return unicodedata.numeric(ch)
        space = context_statespace()
        smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
        cache = space.extra(UnicodeMaskCache)
        if not space.smt_fork(cache.numeric_exists()(smt_codepoint)):
            raise ValueError
        numerator = SymbolicInt(cache.numeric_numerator()(smt_codepoint))
        denominator = SymbolicInt(cache.numeric_denominator()(smt_codepoint))
    return numerator / float(denominator)


def make_registrations():
    register_patch(unicodedata.lookup, with_realized_args(unicodedata.lookup))
    register_patch(unicodedata.name, with_realized_args(unicodedata.name))
    register_patch(unicodedata.decimal, _decimal)
    register_patch(unicodedata.digit, _digit)
    register_patch(unicodedata.numeric, _numeric)
    # TOOD: implement this using get_unicode_categories() - should be easy:
    register_patch(unicodedata.category, with_realized_args(unicodedata.category))
    register_patch(
        unicodedata.bidirectional, with_realized_args(unicodedata.bidirectional)
    )
    register_patch(unicodedata.combining, with_realized_args(unicodedata.combining))
    register_patch(
        unicodedata.east_asian_width, with_realized_args(unicodedata.east_asian_width)
    )
    register_patch(unicodedata.mirrored, with_realized_args(unicodedata.mirrored))
    register_patch(
        unicodedata.decomposition, with_realized_args(unicodedata.decomposition)
    )
    register_patch(unicodedata.normalize, with_realized_args(unicodedata.normalize))
    if sys.version_info >= (3, 8):
        register_patch(
            unicodedata.is_normalized, with_realized_args(unicodedata.is_normalized)
        )
