import operator
import re
import sys
from array import array
from unicodedata import category

if sys.version_info < (3, 11):
    import sre_parse as re_parser
else:
    import re._parser as re_parser

from sys import maxunicode
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import z3  # type: ignore

from crosshair.core import deep_realize, realize, register_patch, with_realized_args
from crosshair.libimpl.builtinslib import AnySymbolicStr, BytesLike, SymbolicInt
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing, ResumedTracing, is_tracing
from crosshair.unicode_categories import CharMask, get_unicode_categories
from crosshair.util import CrossHairInternal, CrossHairValue, debug, is_iterable

ANY = re_parser.ANY
ASSERT = re_parser.ASSERT
ASSERT_NOT = re_parser.ASSERT_NOT
AT = re_parser.AT
AT_BEGINNING = re_parser.AT_BEGINNING
AT_BEGINNING_STRING = re_parser.AT_BEGINNING_STRING
AT_BOUNDARY = re_parser.AT_BOUNDARY
AT_END = re_parser.AT_END
AT_END_STRING = re_parser.AT_END_STRING
AT_NON_BOUNDARY = re_parser.AT_NON_BOUNDARY
BRANCH = re_parser.BRANCH
CATEGORY = re_parser.CATEGORY
CATEGORY_DIGIT = re_parser.CATEGORY_DIGIT
CATEGORY_NOT_DIGIT = re_parser.CATEGORY_NOT_DIGIT
CATEGORY_NOT_SPACE = re_parser.CATEGORY_NOT_SPACE
CATEGORY_NOT_WORD = re_parser.CATEGORY_NOT_WORD
CATEGORY_SPACE = re_parser.CATEGORY_SPACE
CATEGORY_WORD = re_parser.CATEGORY_WORD
IN = re_parser.IN
LITERAL = re_parser.LITERAL
MAX_REPEAT = re_parser.MAX_REPEAT
MAXREPEAT = re_parser.MAXREPEAT
MIN_REPEAT = re_parser.MIN_REPEAT
NEGATE = re_parser.NEGATE
NOT_LITERAL = re_parser.NOT_LITERAL
RANGE = re_parser.RANGE
SUBPATTERN = re_parser.SUBPATTERN
parse = re_parser.parse


class ReUnhandled(Exception):
    pass


_ALL_BYTES_TYPES = (bytes, bytearray, memoryview, array)
_STR_AND_BYTES_TYPES = (str, *_ALL_BYTES_TYPES)
_NO_CHAR = CharMask([])
_ANY_CHAR = CharMask([(0, maxunicode + 1)])
_ANY_NON_NEWLINE_CHAR = _ANY_CHAR.subtract(CharMask([ord("\n")]))
_ASCII_CHAR = CharMask([(0, 128)])
_ASCII_WHITESPACE_CHAR = CharMask([(9, 14), 32])
_UNICODE_WHITESPACE_CHAR = _ASCII_WHITESPACE_CHAR.union(
    CharMask(
        [
            # NOTE: Although 28-31 are in the ASCII range, they only count as whitespace
            # when matching in unicode mode:
            (28, 32),
            133,
            160,
            5760,
            (8192, 8203),
            (8232, 8234),
            8239,
            8287,
            12288,
        ]
    )
)

_CASEABLE_CHARS = None


def caseable_chars():
    global _CASEABLE_CHARS
    if _CASEABLE_CHARS is None:
        codepoints = []
        for i in range(sys.maxunicode + 1):
            ch = chr(i)
            # Exclude the (large) "Other Letter" group that doesn't caseswap:
            if category(ch) in ("Lo"):
                assert ch.casefold() == ch
            else:
                codepoints.append(ch)

        _CASEABLE_CHARS = "".join(codepoints)
    return _CASEABLE_CHARS


_UNICODE_IGNORECASE_MASKS: Dict[int, CharMask] = {}  # codepoint -> CharMask


def unicode_ignorecase_mask(cp: int) -> CharMask:
    mask = _UNICODE_IGNORECASE_MASKS.get(cp)
    if mask is None:
        chars = caseable_chars()
        matches = re.compile(chr(cp), re.IGNORECASE).findall(chars)
        mask = CharMask([ord(c) for c in matches])
        _UNICODE_IGNORECASE_MASKS[cp] = mask
    return mask


def single_char_mask(
    parsed: Tuple[object, Any], flags: int, ord=ord, chr=chr
) -> Optional[CharMask]:
    """
    Compute a CharMask from a parsed regex.

    Takes a pattern object, like those returned by sre_parse.parse().
    Returns None if `parsed` is not a single-character regular expression.
    Returns a list of valid codepoint or codepoint ranges if it can find them, or raises
    ReUnhandled if such an expression cannot be determined.
    """
    (op, arg) = parsed
    isascii = re.ASCII & flags
    if op in (LITERAL, NOT_LITERAL):
        if re.IGNORECASE & flags:
            ret = unicode_ignorecase_mask(arg)
        else:
            ret = CharMask([arg])
        if op is NOT_LITERAL:
            ret = ret.invert()
    elif op is RANGE:
        lo, hi = arg
        if re.IGNORECASE & flags:
            ret = CharMask(
                [
                    # TODO: among other issues, this doesn't handle multi-codepoint caseswaps:
                    (ord(chr(lo).lower()), ord(chr(hi).lower()) + 1),
                    (ord(chr(lo).upper()), ord(chr(hi).upper()) + 1),
                ]
            )
        else:
            ret = CharMask([(lo, hi + 1)])
    elif op is IN:
        ret = CharMask([])
        negate = arg and arg[0][0] is NEGATE
        if negate:
            arg = arg[1:]
        for term in arg:
            submask = single_char_mask(term, flags, ord=ord, chr=chr)
            if submask is None:
                raise ReUnhandled("IN contains non-single-char expression")
            ret = ret.union(submask)
        if negate:
            ret = ret.invert()
    elif op is CATEGORY:
        cats = get_unicode_categories()
        if arg == CATEGORY_DIGIT:
            ret = cats["Nd"]
        elif arg == CATEGORY_NOT_DIGIT:
            ret = cats["Nd"].invert()
        elif arg == CATEGORY_SPACE:
            return _ASCII_WHITESPACE_CHAR if isascii else _UNICODE_WHITESPACE_CHAR
        elif arg == CATEGORY_NOT_SPACE:
            ret = _ASCII_WHITESPACE_CHAR if isascii else _UNICODE_WHITESPACE_CHAR
            return ret.invert()
        elif arg == CATEGORY_WORD:
            ret = cats["word"]
        elif arg == CATEGORY_NOT_WORD:
            ret = cats["word"].invert()
        else:
            raise ReUnhandled("Unsupported category: ", arg)
    elif op is ANY and arg is None:
        # TODO: test dot under ascii mode (seems like we should fall through to the re.ASCII check below)
        return _ANY_CHAR if re.DOTALL & flags else _ANY_NON_NEWLINE_CHAR
    else:
        return None
    if re.ASCII & flags:
        # TODO: this is probably expensive!
        ret = ret.intersect(_ASCII_CHAR)
    return ret


Span = Tuple[int, Union[int, SymbolicInt]]


def _traced_binop(a, op, b):
    if isinstance(a, CrossHairValue) or isinstance(b, CrossHairValue):
        with ResumedTracing():
            return op(a, b)
    return op(a, b)


class _MatchPart:
    def __init__(self, groups: List[Optional[Span]]):
        self._groups = groups

    def _fullspan(self) -> Span:
        span = self._groups[0]
        assert span is not None
        return span

    def _clamp_all_spans(self, start, end):
        groups = self._groups
        for idx, span in enumerate(groups):
            if span is not None:
                (span_start, span_end) = span
                with ResumedTracing():
                    if span_start == span_end:
                        if span_start < start:
                            groups[idx] = (start, start)
                        if span_start > end:
                            groups[idx] = (end, end)

    def isempty(self):
        (start, end) = self._groups[0]
        return _traced_binop(end, operator.le, start)

    def __bool__(self):
        return True

    def __repr__(self):
        return f"<re.Match object; span={self.span()!r}, match={self.group()!r}>"

    def _add_match(self, suffix_match: "_MatchPart") -> "_MatchPart":
        groups: List[Optional[Span]] = [None] * max(
            len(self._groups), len(suffix_match._groups)
        )
        for idx, g in enumerate(self._groups):
            groups[idx] = g
        for idx, g in enumerate(suffix_match._groups):
            if g is not None:
                groups[idx] = g
        my_start = self._fullspan()[0]
        suffix_end = suffix_match._fullspan()[1]
        groups[0] = (my_start, suffix_end)
        return _MatchPart(groups)

    def start(self, group=0):
        return self._groups[group][0]

    def end(self, group=0):
        return self._groups[group][1]

    def span(self, group=0):
        return self._groups[group]


_BACKREF_RE_SOURCE = rb"""
    (?P<prefix> .*?)
    \\
    (?:
        # Note that earlier matches are preferred in regex unions like this:
            (?P<num>        [1-9][0-9]? )    |
        g\< (?P<namednum>  \s*\+?\d+\s* ) \> |
        g\< (?P<named>              \w+ ) \> |
        g\< (?P<namedother>          .* ) \>
    )
    (?P<suffix> .*)
"""
_BACKREF_BYTES_RE = re.compile(_BACKREF_RE_SOURCE, re.VERBOSE | re.MULTILINE)
_BACKREF_STR_RE = re.compile(
    str(_BACKREF_RE_SOURCE, "ascii"), re.VERBOSE | re.MULTILINE
)


class _Match(_MatchPart):
    def __init__(self, groups, pos, endpos, regex, orig_str):
        # fill None in unmatched groups:
        while len(groups) < regex.groups + 1:
            groups.append(None)
        super().__init__(groups)
        self.pos = pos
        if endpos is None:
            with ResumedTracing():
                self.endpos = len(orig_str)
        else:
            self.endpos = endpos
        self.re = regex
        self.string = orig_str

        # Compute lastindex & lastgroup:
        self.lastindex, self.lastgroup = None, None
        _idx_to_name = {num: name for (name, num) in regex.groupindex.items()}
        for idx, grp in enumerate(groups):
            if grp is None:
                continue
            self.lastindex = idx
            if idx in _idx_to_name:
                self.lastgroup = _idx_to_name[idx]

    def __ch_deep_realize__(self, memo):
        # We cannot manually create realistic Match instances.
        # Realize our contents - it's better than nothing
        return _Match(
            deep_realize(self._groups, memo),
            realize(self.pos),
            realize(self.endpos),
            deep_realize(self.re, memo),
            realize(self.string),
        )

    def __getitem__(self, idx):
        return self.group(idx)

    def expand(self, template):
        backref_re = _BACKREF_STR_RE if isinstance(template, str) else _BACKREF_BYTES_RE
        with NoTracing():
            template = realize(template)  # Usually this is a literal string
            match = backref_re.fullmatch(template)
            if match is None:
                return template
            prefix, num, namednum, named, _, suffix = match.groups()
        if num or namednum:
            replacement = self.group(int(num or namednum))
        elif named:
            replacement = self.group(named)
        else:
            raise re.error
        return prefix + replacement + self.expand(suffix)

    def group(self, *nums):
        if not nums:
            nums = (0,)
        ret: List[str] = []
        for num in nums:
            if isinstance(num, str):
                num = self.re.groupindex[num]
            if self._groups[num] is None:
                ret.append(None)
            else:
                start, end = self._groups[num]
                ret.append(self.string[start:end])
        if len(nums) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def groups(self):
        indicies = range(1, len(self._groups))
        if indicies:
            return tuple(self.group(i) for i in indicies)
        else:
            return ()

    def groupdict(self, default=None):
        groups = self._groups
        ret = {}
        for name, idx in self.re.groupindex.items():
            group_range = groups[idx]
            if group_range is not None:
                ret[name] = group_range
        return ret


_REMOVE = object()


def _patt_replace(list_tree: List, from_obj: object, to_obj: object = _REMOVE) -> List:
    """
    >>> _patt_replace([[], [2, None]], None, 3)
    [[], [2, 3]]
    >>> _patt_replace([[], [None, 7]], None, _REMOVE)
    [[], [7]]
    """
    for idx, child in enumerate(list_tree):
        if child is from_obj:
            if to_obj is _REMOVE:
                return list_tree[:idx] + list_tree[idx + 1 :]
            else:
                return [(to_obj if x is from_obj else x) for x in list_tree]
        if not is_iterable(child):
            continue
        newchild = _patt_replace(child, from_obj, to_obj)
        if newchild is not child:
            # Found it; make a copy of this list with the new item:
            newlist = list(list_tree)
            newlist[idx] = newchild
            return newlist
    # nothing changed; re-use the original list
    return list_tree


_END_GROUP_MARKER = object()


def _internal_match_patterns(
    top_patterns: Any,
    flags: int,
    string: AnySymbolicStr,
    offset: int,
    allow_empty: bool = True,
    ord=ord,
    chr=chr,
) -> Optional[_MatchPart]:
    """
    >>> import sre_parse
    >>> from crosshair.core_and_libs import standalone_statespace, NoTracing
    >>> from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
    >>> with standalone_statespace, NoTracing():
    ...     string = LazyIntSymbolicStr(list(map(ord, 'aabb')))
    ...     _internal_match_patterns(sre_parse.parse('a+'), 0, string, 0).span()
    ...     _internal_match_patterns(sre_parse.parse('ab'), 0, string, 1).span()
    (0, 2)
    (1, 3)
    """
    space = context_statespace()
    with ResumedTracing():
        matchablestr = string[offset:] if offset > 0 else string

    if len(top_patterns) == 0:
        return _MatchPart([(offset, offset)]) if allow_empty else None
    pattern = top_patterns[0]

    def continue_matching(prefix):
        sub_allow_empty = allow_empty if prefix.isempty() else True
        suffix = _internal_match_patterns(
            top_patterns[1:],
            flags,
            string,
            prefix.end(),
            sub_allow_empty,
            ord=ord,
            chr=chr,
        )
        if suffix is None:
            return None
        return prefix._add_match(suffix)

    # TODO: using a typed internal function triggers __hash__es inside the typing module.
    # Seems like this casues nondeterminism due to a global LRU cache used by the typing module.
    def fork_on(expr, sz):
        if space.smt_fork(expr):
            return continue_matching(
                _MatchPart([(offset, _traced_binop(offset, operator.add, sz))])
            )
        else:
            return None

    mask = single_char_mask(pattern, flags, ord=ord, chr=chr)
    if mask is not None:
        with ResumedTracing():
            if any([offset < 0, offset >= len(string)]):
                return None
            char = ord(string[offset])
        if isinstance(char, int):  # Concrete int? Just check it!
            if mask.covers(char):
                return continue_matching(
                    _MatchPart([(offset, _traced_binop(offset, operator.add, 1))])
                )
            else:
                return None
        smt_ch = SymbolicInt._coerce_to_smt_sort(char)
        return fork_on(mask.smt_matches(smt_ch), 1)

    (op, arg) = pattern
    if op in (MIN_REPEAT, MAX_REPEAT):
        (min_repeat, max_repeat, subpattern) = arg
        if max_repeat < min_repeat:
            return None
        reps = 0
        overall_match = _MatchPart([(offset, offset)])
        while reps < min_repeat:
            submatch = _internal_match_patterns(
                subpattern, flags, string, overall_match.end(), True, ord=ord, chr=chr
            )
            if submatch is None:
                return None
            overall_match = overall_match._add_match(submatch)
            reps += 1
        if max_repeat != MAXREPEAT and reps >= max_repeat:
            return continue_matching(overall_match)

        if max_repeat == MAXREPEAT:
            remaining_reps = max_repeat
        else:
            remaining_reps = max_repeat - min_repeat

        if op is MIN_REPEAT:
            # Non-greedy match: try the shortest possible match first.
            short_match = continue_matching(overall_match)
            if short_match is not None:
                return short_match

        remaining_matcher = _patt_replace(
            top_patterns, arg, (1, remaining_reps, subpattern)
        )
        remainder_allow_empty = allow_empty or not overall_match.isempty()
        remainder_match = _internal_match_patterns(
            remaining_matcher,
            flags,
            string,
            overall_match.end(),
            remainder_allow_empty,
            ord=ord,
            chr=chr,
        )
        if remainder_match is not None:
            return overall_match._add_match(remainder_match)

        if op is MAX_REPEAT:
            # Greedy match: didn't match more repetitions - try from here.
            return continue_matching(overall_match)

        return None
    elif op is BRANCH and arg[0] is None:
        # NOTE: order matters - earlier branches are more greedily matched than later branches.
        branches = arg[1]
        first_path = list(branches[0]) + list(top_patterns)[1:]
        submatch = _internal_match_patterns(
            first_path, flags, string, offset, allow_empty, ord=ord, chr=chr
        )
        if submatch is not None:
            return submatch
        if len(branches) <= 1:
            return None
        else:
            return _internal_match_patterns(
                _patt_replace(top_patterns, branches, branches[1:]),
                flags,
                string,
                offset,
                allow_empty,
                ord=ord,
                chr=chr,
            )
    elif op is AT:
        if arg in (AT_BEGINNING, AT_BEGINNING_STRING):
            begins_string = fork_on(SymbolicInt._coerce_to_smt_sort(offset) == 0, 0)
            if begins_string:
                return begins_string
            if arg is AT_BEGINNING and re.MULTILINE & flags:
                with ResumedTracing():
                    prev_char = ord(string[offset - 1])
                return fork_on(
                    SymbolicInt._coerce_to_smt_sort(prev_char) == ord("\n"), 0
                )
            return None
        with ResumedTracing():
            matchable_len = len(matchablestr)
        ends_string = space.smt_fork(
            SymbolicInt._coerce_to_smt_sort(matchable_len) == 0
        )
        if arg in (AT_END, AT_END_STRING):
            if ends_string:
                return continue_matching(_MatchPart([(offset, offset)]))
            if arg is AT_END and re.MULTILINE & flags:
                with ResumedTracing():
                    next_char = ord(string[offset])
                return fork_on(
                    SymbolicInt._coerce_to_smt_sort(next_char) == ord("\n"), 0
                )
            return None
        elif arg in (AT_BOUNDARY, AT_NON_BOUNDARY):
            if ends_string or offset == 0:
                if arg == AT_BOUNDARY:
                    return continue_matching(_MatchPart([(offset, offset)]))
                else:
                    assert arg == AT_NON_BOUNDARY
                    return None
            with ResumedTracing():
                left = ord(string[offset - 1])
                right = ord(string[offset])
            wordmask = get_unicode_categories()["word"]
            left_expr = wordmask.smt_matches(SymbolicInt._coerce_to_smt_sort(left))
            right_expr = wordmask.smt_matches(SymbolicInt._coerce_to_smt_sort(right))
            at_boundary_expr = z3.Xor(left_expr, right_expr)
            if arg == AT_NON_BOUNDARY:
                at_boundary_expr = z3.Not(at_boundary_expr)
            return fork_on(at_boundary_expr, 0)
    elif op in (ASSERT, ASSERT_NOT):
        (direction_int, subpattern) = arg
        positive_look = op == ASSERT
        if direction_int == 1:
            matched = _internal_match_patterns(
                subpattern, flags, string, offset, True, ord=ord, chr=chr
            )
        else:
            assert direction_int == -1
            minwidth, maxwidth = subpattern.getwidth()
            if minwidth != maxwidth:
                raise re.error("")
            rewound = offset - minwidth
            if rewound < 0:
                return None
            matched = _internal_match_patterns(
                subpattern, flags, string, rewound, True, ord=ord, chr=chr
            )
        if bool(matched) != bool(positive_look):
            return None
        return _internal_match_patterns(
            top_patterns[1:], flags, string, offset, allow_empty, ord=ord, chr=chr
        )
    elif op is SUBPATTERN:
        (groupnum, _a, _b, subpatterns) = arg
        if (_a, _b) != (0, 0):
            raise ReUnhandled("unsupported subpattern args")
        new_top = (
            list(subpatterns)
            + [(_END_GROUP_MARKER, (groupnum, offset))]
            + list(top_patterns)[1:]
        )
        return _internal_match_patterns(
            new_top, flags, string, offset, allow_empty, ord=ord, chr=chr
        )
    elif op is _END_GROUP_MARKER:
        (group_num, begin) = arg
        match = continue_matching(_MatchPart([(offset, offset)]))
        if match is None:
            return None
        while len(match._groups) <= group_num:
            match._groups.append(None)
        match._groups[group_num] = (begin, offset)
        return match
    raise ReUnhandled(op)


def _match_pattern(
    compiled_regex: re.Pattern,
    orig_str: Union[AnySymbolicStr, BytesLike],
    pos: int,
    endpos: Optional[int] = None,
    subpattern: Optional[List] = None,
    allow_empty=True,
    ord=ord,
    chr=chr,
) -> Optional[_Match]:
    assert not is_tracing()
    if subpattern is None:
        subpattern = cast(List, parse(compiled_regex.pattern, compiled_regex.flags))
    with ResumedTracing():
        trimmed_str = orig_str[:endpos]
    matchpart = _internal_match_patterns(
        subpattern,
        compiled_regex.flags,
        trimmed_str,
        pos,
        allow_empty,
        ord=ord,
        chr=chr,
    )
    if matchpart is None:
        return None
    match_start, match_end = matchpart._fullspan()
    if _traced_binop(match_start, operator.eq, match_end):
        matchpart._clamp_all_spans(0, len(orig_str))
    return _Match(matchpart._groups, pos, endpos, compiled_regex, orig_str)


def _compile(*a):
    # Symbolic regexes aren't supported, and it's expensive to perform compilation
    # with tracing enabled.
    with NoTracing():
        return re._compile(*deep_realize(a))


def _check_str_or_bytes(patt: re.Pattern, obj: Any):
    if not isinstance(patt, re.Pattern):
        raise TypeError  # TODO: e.g. "descriptor 'search' for 're.Pattern' objects doesn't apply to a 'str' object"
    if not isinstance(obj, _STR_AND_BYTES_TYPES):
        raise TypeError(f"expected string or bytes-like object, got '{type(obj)}'")
    if isinstance(patt.pattern, str):
        if isinstance(obj, str):
            return (chr, ord)
        raise TypeError("cannot use a bytes pattern on a string-like object")
    else:
        if isinstance(obj, _ALL_BYTES_TYPES):
            return (lambda i: bytes([i]), lambda i: i)
        raise TypeError("cannot use a string pattern on a bytes-like object")


def _finditer_symbolic(
    patt: re.Pattern, string: AnySymbolicStr, pos: int, endpos: int, chr=chr, ord=ord
) -> Iterable[_Match]:
    last_match_was_empty = False
    while True:
        with NoTracing():
            if pos > endpos:
                break
            allow_empty = not last_match_was_empty
            match = _match_pattern(
                patt, string, pos, endpos, allow_empty=allow_empty, chr=chr, ord=ord
            )
            last_match_was_empty = False
            if not match:
                pos += 1
                continue
        yield match
        with NoTracing():
            if match.start() == match.end():
                if not allow_empty:
                    raise CrossHairInternal("Unexpected empty match")
                last_match_was_empty = True
            else:
                pos = match.end()


def _finditer(
    self: re.Pattern,
    string: Union[str, AnySymbolicStr, bytes],
    pos: int = 0,
    endpos: Optional[int] = None,
) -> Iterable[Union[re.Match, _Match]]:
    chr, ord = _check_str_or_bytes(self, string)
    if not isinstance(pos, int):
        raise TypeError
    if not (endpos is None or isinstance(endpos, int)):
        raise TypeError
    pos, endpos = realize(pos), realize(endpos)
    strlen = len(string)
    with NoTracing():
        if isinstance(string, AnySymbolicStr):
            pos, endpos, _ = slice(pos, endpos, 1).indices(realize(strlen))
            with ResumedTracing():
                try:
                    yield from _finditer_symbolic(
                        self, string, pos, endpos, chr=chr, ord=ord
                    )
                    return
                except ReUnhandled as e:
                    debug("Unsupported symbolic regex", self.pattern, e)
    if endpos is None:
        yield from re.Pattern.finditer(self, realize(string), pos)
    else:
        yield from re.Pattern.finditer(self, realize(string), pos, endpos)


def _fullmatch(
    self: re.Pattern, string: Union[str, AnySymbolicStr, bytes], pos=0, endpos=None
):
    with NoTracing():
        if isinstance(string, (AnySymbolicStr, BytesLike)):
            with ResumedTracing():
                chr, ord = _check_str_or_bytes(self, string)
            try:
                compiled = cast(List, parse(self.pattern, self.flags))
                compiled.append((AT, AT_END_STRING))
                return _match_pattern(
                    self, string, pos, endpos, compiled, chr=chr, ord=ord
                )
            except ReUnhandled as e:
                debug("Unsupported symbolic regex", self.pattern, e)
        if endpos is None:
            return re.Pattern.fullmatch(self, realize(string), pos)
        else:
            return re.Pattern.fullmatch(self, realize(string), pos, endpos)


def _match(
    self, string: Union[str, AnySymbolicStr], pos=0, endpos=None
) -> Union[None, re.Match, _Match]:
    with NoTracing():
        if isinstance(string, (AnySymbolicStr, BytesLike)):
            with ResumedTracing():
                chr, ord = _check_str_or_bytes(self, string)
            try:
                return _match_pattern(self, string, pos, endpos, chr=chr, ord=ord)
            except ReUnhandled as e:
                debug("Unsupported symbolic regex", self.pattern, e)
        if endpos is None:
            return re.Pattern.match(self, realize(string), pos)
        else:
            return re.Pattern.match(self, realize(string), pos, endpos)


def _search(
    self: re.Pattern,
    string: Union[str, AnySymbolicStr, bytes],
    pos: int = 0,
    endpos: Optional[int] = None,
) -> Union[None, re.Match, _Match]:
    chr, ord = _check_str_or_bytes(self, string)
    if not isinstance(pos, int):
        raise TypeError
    if not (endpos is None or isinstance(endpos, int)):
        raise TypeError
    pos, endpos = realize(pos), realize(endpos)
    mylen = string.__len__()
    with NoTracing():
        if isinstance(string, (AnySymbolicStr, BytesLike)):
            pos, endpos, _ = slice(pos, endpos, 1).indices(realize(mylen))
            try:
                while pos < endpos:
                    match = _match_pattern(self, string, pos, endpos, chr=chr, ord=ord)
                    if match:
                        return match
                    pos += 1
                return None
            except ReUnhandled as e:
                debug("Unsupported symbolic regex", self.pattern, e)
        if endpos is None:
            return re.Pattern.search(self, realize(string), pos)
        else:
            return re.Pattern.search(self, realize(string), pos, endpos)


def _sub(self, repl, string, count=0):
    (result, _) = _subn(self, repl, string, count)
    return result


def _subn(
    self: re.Pattern, repl: Union[str, Callable], string: str, count: int = 0
) -> Tuple[str, int]:
    if not isinstance(self, re.Pattern):
        raise TypeError
    if isinstance(repl, _STR_AND_BYTES_TYPES):
        _check_str_or_bytes(self, repl)

        def replfn(m):
            return m.expand(repl)

    elif callable(repl):
        replfn = repl
    else:
        raise TypeError
    _check_str_or_bytes(self, string)
    if not isinstance(count, int):
        raise TypeError
    match = self.search(string)
    if match is None:
        return (string, 0)
    result_prefix = string[: match.start()] + replfn(match)
    if count == 1:
        return (result_prefix + string[match.end() :], 1)
    if match.end() == match.start():
        remaining = string[match.end() + 1 :]
    else:
        remaining = string[match.end() :]
    (result_suffix, suffix_replacements) = _subn(self, repl, remaining, count - 1)
    return (result_prefix + result_suffix, suffix_replacements + 1)


def make_registrations():
    register_patch(re._compile, _compile)
    register_patch(re.Pattern.search, _search)
    register_patch(re.Pattern.match, _match)
    register_patch(re.Pattern.fullmatch, _fullmatch)
    register_patch(re.Pattern.split, with_realized_args(re.Pattern.split))
    register_patch(re.Pattern.findall, with_realized_args(re.Pattern.findall))
    register_patch(re.Pattern.finditer, _finditer)
    register_patch(re.Pattern.sub, _sub)
    register_patch(re.Pattern.subn, _subn)
