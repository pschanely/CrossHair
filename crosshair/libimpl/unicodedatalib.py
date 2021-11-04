import unicodedata

from crosshair.core import register_patch
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
    register_patch(unicodedata.decimal, _decimal)
    register_patch(unicodedata.digit, _digit)
    register_patch(unicodedata.numeric, _numeric)
