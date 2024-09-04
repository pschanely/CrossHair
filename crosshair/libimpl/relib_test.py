import random
import re
from typing import Optional

import pytest

from crosshair import ResumedTracing
from crosshair.core import deep_realize, proxy_for_type
from crosshair.core_and_libs import NoTracing, standalone_statespace
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr, SymbolicBytes
from crosshair.libimpl.relib import _BACKREF_STR_RE, _match_pattern
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import CANNOT_CONFIRM, CONFIRMED, POST_FAIL, MessageType
from crosshair.test_util import check_states
from crosshair.util import CrossHairInternal


def eval_regex(re_string, flags, test_string, offset, endpos=None):
    py_patt = re.compile(re_string, flags)
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr([ord(c) for c in test_string])
            match = _match_pattern(py_patt, s, offset, endpos)
        return deep_realize(match)


def test_handle_simple():
    assert eval_regex("abc", 0, "abc", 0) is not None
    assert eval_regex("abc", 0, "ab", 0) is None


def test_handle_or():
    assert eval_regex("a|bc", 0, "bc", 0) is not None
    assert eval_regex("a|bc", 0, "bc", 0).span() == (0, 2)

    assert eval_regex("a|bc", 0, "ab", 0) is not None
    assert eval_regex("a|bc", 0, "ab", 0).span() == (0, 1)

    assert eval_regex("a|bc", 0, "c", 0) is None
    assert eval_regex("a|bc", 0, "bd", 0) is None


def test_handle_start_markers():
    assert eval_regex(r"^ab", 0, "abc", 0) is not None
    assert eval_regex(r"\Aab", 0, "abc", 0) is not None
    assert eval_regex(r"^", 0, "", 0) is not None
    # Surprisingly!: re.compile('^bc').match('abc', 1) is None
    # Even more surprisingly, the end markers are happy to match off of endpos.
    assert eval_regex(r"^bc", 0, "abc", 1) is None
    assert eval_regex(r"^bc", re.MULTILINE, "a\nbc", 2) is not None
    assert eval_regex(r"^bc", 0, "a\nbc", 2) is None


def test_handle_end_markers():
    assert eval_regex(r"abc$", 0, "abc", 0) is not None
    assert eval_regex(r"abc$", 0, "abcd", 0, 3) is not None
    assert eval_regex(r"abc\Z", 0, "abc", 0) is not None
    assert eval_regex(r"abc\Z", re.MULTILINE, "abc", 0) is not None
    assert eval_regex("abc$", re.MULTILINE, "abc\n", 0) is not None
    assert eval_regex("a$.b", re.MULTILINE | re.DOTALL, "a\nb", 0) is not None
    assert eval_regex("abc$", 0, "abc\n", 0) is None
    assert eval_regex("abc$", re.MULTILINE, "abcd", 0) is None


def test_handle_range():
    assert eval_regex("[a-z]7", 0, "b7", 0) is not None
    assert eval_regex("[a-z]7", 0, "z7", 0) is not None
    assert eval_regex("[a-z]7", 0, "A7", 0) is None


def test_handle_sets():
    assert eval_regex("[a7]", 0, "7", 0) is not None
    assert eval_regex("[^a7]", 0, "7", 0) is None
    assert eval_regex("[^3-9]", 0, "7", 0) is None
    assert eval_regex("[^a]", 0, "7", 0) is not None
    assert eval_regex("[^a]", 0, "a", 0) is None
    assert eval_regex("[^a]", 0, "", 0) is None


def test_handle_ascii_wildcard():
    assert eval_regex("1.2", re.A, "1x2", 0) is not None
    assert eval_regex("1.2", re.A, "1\x002", 0) is not None
    assert eval_regex("1.2", re.A, "111", 0) is None


def test_handle_repeats():
    assert eval_regex("a+a", 0, "aa", 0) is not None
    assert eval_regex("s", 0, "ssss", 0).span() == (0, 1)
    assert eval_regex("ss", 0, "ssss", 0).span() == (0, 2)
    assert eval_regex("s{1,2}x", 0, "sx", 0) is not None
    assert eval_regex("s{1,2}x", 0, "ssx", 0) is not None
    assert eval_regex("s{1,2}x", 0, "sssx", 0) is None
    assert eval_regex("s{1,2}x", 0, "x", 0) is None
    assert eval_regex("s{2,3}", 0, "ssss", 0) is not None
    assert eval_regex("s{2,3}", 0, "ssss", 0).span() == (0, 3)
    assert eval_regex("y*", 0, "y", 0) is not None
    assert eval_regex("y*", 0, "y", 0).span() == (0, 1)
    assert eval_regex("y*e+", 0, "ye", 0) is not None
    assert eval_regex("y*e", 0, "yye", 0) is not None
    assert eval_regex("y*e", 0, "yye", 0).span() == (0, 3)
    assert eval_regex("y*e+s{2,3}x", 0, "yessx", 0) is not None
    assert eval_regex("y*e+s{2,3}x", 0, "essx", 0) is not None
    assert eval_regex("y*e+s{2,3}x", 0, "yyessssx", 0) is None
    assert eval_regex("y*e+s{2,3}x", 0, "yssx", 0) is None
    assert eval_regex("y*e+s{2,3}x", 0, "ex", 0) is None


def test_handle_nongreedy_repeats():
    assert eval_regex("a+?", 0, "a", 0) is not None
    assert eval_regex("a+?b", 0, "aab", 0) is not None
    assert eval_regex("a*?", 0, "aa", 0).span() == (0, 0)
    assert eval_regex("a*?b", 0, "aabaa", 0).span() == (0, 3)


def test_handle_ascii_numeric():
    assert eval_regex(r"a\d", re.A, "a3", 0) is not None
    assert eval_regex(r"a\d", re.A, "a0", 0) is not None
    assert eval_regex(r"a\d", re.A, "a-", 0) is None


def test_handle_ascii_whitespace():
    assert eval_regex(r"a\s", re.A, "a ", 0) is not None
    assert eval_regex(r"a\s", re.A, "a\r", 0) is not None
    assert eval_regex(r"a\s", re.A, "a.", 0) is None
    assert eval_regex(r"a\s", re.A, "a\x1c", 0) is None
    assert eval_regex(r"a\s", 0, "a\x1c", 0) is not None


def test_word_boundaries():
    assert eval_regex(r".\b", 0, "a", 0) is not None
    assert eval_regex(r".\b", 0, "a ", 0) is not None
    assert eval_regex(r"\b.", 0, "a", 0) is not None
    assert eval_regex(r".\b", 0, "ab", 0) is None
    assert eval_regex(r"\b.", 0, "", 0) is None


def test_word_non_boundaries():
    assert eval_regex(r"\B", 0, "ab", 1) is not None
    assert eval_regex(r"\B", 0, "ab", 0) is None
    assert eval_regex(r"\B", 0, "ab", 2) is None
    assert eval_regex(r"a\B.", 0, "ab", 0) is not None
    assert eval_regex(r"a\B.", 0, "a ", 0) is None


def test_handle_noncapturing_group():
    assert eval_regex("(?:a|b)c", 0, "ac", 0) is not None
    assert eval_regex("(?:a|b)c", 0, "bc", 0) is not None
    assert eval_regex("(?:a|b)c", 0, "a", 0) is None


def test_handle_capturing_group():
    assert eval_regex("(a|b)c", 0, "ac", 0) is not None
    assert eval_regex("(a|b)c", 0, "a", 0) is None
    assert type(eval_regex("(a|b)c", 0, "bc", 0).groups()[0]) == str
    assert eval_regex("(a|b)c", 0, "bc", 0).groups() == ("b",)


def test_handle_named_groups():
    assert eval_regex("(?P<foo>a|b)c", 0, "bc", 0) is not None
    assert eval_regex("(?P<foo>a|b)c", 0, "bc", 0)["foo"] == "b"


def test_handle_optional_named_groups():
    assert eval_regex("a(?P<foo>b)?", 0, "a", 0)["foo"] is None
    assert eval_regex("a(?P<foo>b)?c", 0, "ac", 0)["foo"] is None


def test_handle_nested_groups():
    assert eval_regex("(a|b(xx))+(c)?", 0, "bxxc", 0) is not None
    assert eval_regex("(bxx)(c)?", 0, "bxxc", 0).groups() == ("bxx", "c")
    assert eval_regex("(a|b(xx))+(c)?", 0, "bxxc", 0).groups() == ("bxx", "xx", "c")
    assert eval_regex("(a|b(xx))+(c)?", 0, "a", 0).groups() == ("a", None, None)


def test_with_fuzzed_inputs() -> None:
    rand = random.Random(253209)

    def check(pattern, literal_string, offset):
        flags = re.ASCII | re.DOTALL
        sym_match = eval_regex(pattern, flags, literal_string, offset)
        py_match = re.compile(pattern, flags).match(literal_string, offset)
        if (sym_match is None) != (py_match is None):
            assert py_match == sym_match
        if py_match is None:
            return
        assert py_match.span() == sym_match.span()
        assert py_match.group(0) == sym_match.group(0)
        assert py_match.groups() == sym_match.groups()
        assert py_match.pos == sym_match.pos
        assert py_match.endpos == sym_match.endpos
        assert py_match.lastgroup == sym_match.lastgroup

    for iter in range(100):
        literal_string = "".join(
            rand.choice(["a", "5", "_"])
            for _ in range(rand.choice([0, 1, 1, 2, 2, 3, 4]))
        )
        pattern = "".join(
            rand.choice(["a", "5", "."]) + rand.choice(["", "", "+", "*"])
            for _ in range(rand.choice([0, 1, 1, 2, 2]))
        )
        offset = rand.choice([0, 0, 0, 0, 1])
        check(pattern, literal_string, offset)


def test_fullmatch_basic_fail() -> None:
    def f(s: str) -> Optional[re.Match]:
        """post: _"""
        return re.compile("a").fullmatch(s)

    check_states(f, POST_FAIL)


def test_star_fail() -> None:
    def f(s: str) -> bool:
        """
        pre: s[1] == 'a'
        post: _
        """
        return not re.fullmatch("a*", s)

    check_states(f, POST_FAIL)


def test_plus_unknown() -> None:
    def f(s: str) -> bool:
        """
        pre: len(s) > 0
        post: _
        """
        return bool(re.fullmatch(".+", s, re.DOTALL))

    check_states(f, CANNOT_CONFIRM)


def test_greedy_backtracking() -> None:
    def f(s: str) -> int:
        """
        pre: len(s) == 3
        post: _ == 3
        """
        return re.match(".+.", s, re.A | re.DOTALL).end()  # type: ignore

    check_states(f, CONFIRMED)


def test_fullmatch_basic_ok() -> None:
    def f(s: str) -> Optional[re.Match]:
        """
        pre: s == 'a'
        post: _
        """
        return re.compile("a").fullmatch(s)

    check_states(f, CONFIRMED)


def test_fullmatch_matches_whole_string() -> None:
    def f(s: str) -> Optional[re.Match]:
        """
        pre: len(s) == 3
        post: implies(_, s[-1] == 'b')
        """
        return re.compile("a+b+").fullmatch(s)

    check_states(f, CONFIRMED)


def test_match_properties() -> None:
    match = re.compile("(a)b").match("01ab9", 2, 4)

    # Before we begin, quickly double-check that our expectations match what Python
    # actually does:
    assert match is not None
    assert match.span() == (2, 4)
    assert match.groups() == ("a",)
    assert match.group(0) == "ab"
    assert match.group(1) == "a"
    assert match[0] == "ab"
    assert match.pos == 2
    assert match.endpos == 4
    assert match.lastgroup is None
    assert match.string == "01ab9"
    assert match.re.pattern == "(a)b"
    assert match.expand(r"z\1z") == "zaz"
    assert match.groupdict() == {}
    assert match.start(1) == 2
    assert match.lastindex == 1

    def f(s: str) -> Optional[re.Match]:
        r"""
        pre: s == '01ab9'
        post: _.span() == (2, 4)
        post: _.groups() == ('a',)
        post: _.group(0) == 'ab'
        post: _.group(1) == 'a'
        post: _[0] == 'ab'
        post: _.pos == 2
        post: _.endpos == 4
        post: _.lastgroup == None
        post: _.string == '01ab9'
        post: _.re.pattern == '(a)b'
        post: _.expand(r'z\1z') == 'zaz'
        post: _.groupdict() == {}
        post: _.start(1) == 2
        post: _.lastindex == 1
        """
        return re.compile("(a)b").match(s, 2, 4)

    check_states(f, CONFIRMED)


def test_fullmatch_complex_fail() -> None:
    def f(s: str) -> str:
        """
        pre: re.fullmatch('a+Xb', s)
        post: _ != 'X'
        """
        return s[2]

    check_states(f, MessageType.POST_FAIL)


@pytest.mark.demo
def test_match() -> None:
    def f(s: str) -> Optional[re.Match]:
        """
        Can the captured character in this regex be "x"?

        NOTE: Although this use case is solved quickly, many regex problems will
        require a few minutes of processing time or more.

        post: _ is None or _.group(1) != "x"
        """
        return re.compile("a([a-z])").match(s)

    check_states(f, POST_FAIL)


def test_match_basic_fail2() -> None:
    def f(s: str) -> bool:
        """post: implies(_, len(s) <= 3)"""
        return bool(re.compile("ab?c").match(s))

    check_states(f, POST_FAIL)


def test_number_parse() -> None:
    number_re = re.compile(r"(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?")

    def f(s: str):
        """
        pre: len(s) == 4
        post: not _
        """
        return bool(number_re.fullmatch(s))

    check_states(
        f,
        POST_FAIL,
        AnalysisOptionSet(max_iterations=20),
    )


def test_lookbehind() -> None:
    regex = re.compile(r"(?<=a)bc")

    def f(s: str):
        """
        post: not _
        """
        return bool(regex.search(s))

    check_states(f, POST_FAIL)


def test_backref_re():
    assert _BACKREF_STR_RE.fullmatch(r"\1").group("num") == "1"
    assert _BACKREF_STR_RE.fullmatch(r"ab\1cd").group("num") == "1"
    assert _BACKREF_STR_RE.fullmatch(r"$%^ \g<_cat> &*").group("named") == "_cat"
    assert _BACKREF_STR_RE.fullmatch(r"\g< cat>").group("namedother") == " cat"
    assert _BACKREF_STR_RE.fullmatch(r"\g<0>").group("namednum") == "0"
    assert _BACKREF_STR_RE.fullmatch(r"\g<+100>").group("namednum") == "+100"
    assert _BACKREF_STR_RE.fullmatch(r"\1 foo \2").group("num") == "1"

    # "\g<0>" is OK; "\0" is not:
    assert _BACKREF_STR_RE.fullmatch(r"\g<0>")
    assert not _BACKREF_STR_RE.fullmatch(r"\0")


def test_template_expansion():
    regex = re.compile("(a)(?P<foo>b)")
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr(list(map(ord, "abc")))
        match = regex.match(s)
        assert match.expand(r"z\1z") == "zaz"
        assert match.expand(r"z\g<foo>z") == "zbz"
        assert match.expand(r"z\g<0>z") == "zabz"
        assert match.expand(r"\1z\1\1") == "azaa"


def test_finditer():
    regex = re.compile("a")
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr(list(map(ord, "abaa")))
        itr = regex.finditer(s)
        assert next(itr).pos == 0
        assert next(itr).pos == 2
        assert next(itr).pos == 3
        try:
            unexpected_match = next(itr)
            assert False, unexpected_match
        except StopIteration:
            pass


def test_charmatch_literal_does_not_fork():
    letters = re.compile("[a-z]")
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr(list(map(ord, "abaa")))

            def explode(*a, **kw):
                raise CrossHairInternal

            space.smt_fork = explode
        match = letters.match(s)
        assert match
        assert match.group(0) == "a"


def test_symbolic_offset():
    _all_zeros = re.compile("0*$")
    with standalone_statespace as space:
        with NoTracing():
            string = LazyIntSymbolicStr(list(map(ord, "21000")))
            offset = proxy_for_type(int, "offset")
            endpos = proxy_for_type(int, "endpos")
        space.add(offset == 2)
        space.add(endpos == 5)
        assert _all_zeros.match(string, offset)
        assert not _all_zeros.match(string, offset - 1)
        assert not _all_zeros.match(string + "1", offset)
        assert _all_zeros.match(string + "1", offset, endpos)
        assert not _all_zeros.match(string + "1", offset, endpos + 1)


@pytest.mark.parametrize(
    "patt_char,match_char",
    [
        ("ß", "ẞ"),
        ("ẞ", "ß"),
        ("İ", "i"),
        ("i", "İ"),
        ("Ⓐ", "ⓐ"),
        ("ⓐ", "Ⓐ"),
    ],
)
def test_ignorecase_matches(space, patt_char, match_char):
    pattern = re.compile(patt_char, re.IGNORECASE)
    # sanity check that regular python does what we expect:
    assert pattern.fullmatch(match_char)
    symbolic_match_char = LazyIntSymbolicStr(list(map(ord, match_char)))
    with ResumedTracing():
        assert pattern.fullmatch(symbolic_match_char)


@pytest.mark.parametrize(
    "patt_char,match_char",
    [
        ("a", "ⓐ"),
        ("ß".upper(), "ß"),
        ("ß", "ß".upper()),
        ("İ".lower(), "İ"),
        ("İ", "İ".lower()),
    ],
)
def test_ignorecase_nonmatches(space, patt_char, match_char):
    pattern = re.compile(patt_char, re.IGNORECASE)
    # sanity check that regular python does what we expect:
    assert not pattern.fullmatch(match_char)
    symbolic_match_char = LazyIntSymbolicStr(list(map(ord, match_char)))
    with ResumedTracing():
        assert not pattern.fullmatch(symbolic_match_char)


def test_bytes_based_pattern(space):
    string = SymbolicBytes(b"abbc")
    with ResumedTracing():
        assert re.fullmatch(b"ab+c", string)
        assert [m.span() for m in re.finditer(b"b", string)] == [(1, 2), (2, 3)]
