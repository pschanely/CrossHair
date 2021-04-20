import re
from typing import *

from sre_parse import ANY, AT, BRANCH, IN, LITERAL, RANGE, SUBPATTERN  # type: ignore
from sre_parse import MAX_REPEAT, MAXREPEAT  # type: ignore
from sre_parse import CATEGORY, CATEGORY_DIGIT  # type: ignore
from sre_parse import AT_END, AT_END_STRING  # type: ignore
from sre_parse import parse


import z3  # type: ignore

from crosshair import debug, register_patch, StateSpace
from crosshair import realize, with_realized_args, IgnoreAttempt

from crosshair.libimpl.builtinslib import SymbolicInt, SymbolicStr
from crosshair.util import is_iterable
from crosshair.util import CrosshairUnsupported

# TODO: test _Match methods
# TODO: SUBPATTERN
# TODO: CATEGORY
# TODO: re.MULTILINE
# TODO: re.DOTALL
# TODO: re.IGNORECASE
# TODO: Give up on re.LOCALE
# TODO: bytes input and re.ASCII
# TODO: Match edge conditions; IndexError etc
# TODO: Match.__repr__
# TODO: wait for unicode support in z3 and adapt this.
# TODO: greediness by default; also nongreedy: +? *? ?? {n,m}?
# TODO: ATs: parse(r'\A^\b\B$\Z', re.MULTILINE) == [(AT, AT_BEGINNING_STRING),
#         (AT, AT_BEGINNING), (AT, AT_BOUNDARY), (AT, AT_NON_BOUNDARY),
#         (AT, AT_END), (AT, AT_END_STRING)]
# TODO: capture groups
# TODO: backreferences to capture groups: parse(r'(\w) \1') ==
#         [(SUBPATTERN, (1, 0, 0, [(IN, [(CATEGORY, CATEGORY_WORD)])])),
#          (LITERAL, 32), (GROUPREF, 1)]
# TODO: categories: CATEGORY_SPACE, CATEGORY_WORD, CATEGORY_LINEBREAK
# TODO: NEGATE: parse(r'[^34]') == [(IN, [(NEGATE, None), (LITERAL, 51), (LITERAL, 52)])]
# TODO: NOT_LITERAL: parse(r'[^\n]') == [(NOT_LITERAL, 10)]
# TODO: search()
# TODO: split()
# TODO: findall() and finditer()
# TODO: sub() and subn()
# TODO: positive/negative lookahead/lookbehind


class ReUnhandled(Exception):
    pass


def single_char_regex(parsed: Tuple[object, Any], flags: int) -> Optional[z3.ExprRef]:
    """
    Takes a pattern object, like those returned by sre_parse.parse().
    Returns None if `parsed` is not a single-character regular expression.
    Returns an equivalent z3 regular expression if it can find one, or raises
    ReUnhandled if such an expression cannot be determined.
    """
    (op, arg) = parsed
    if op is LITERAL:
        if re.IGNORECASE & flags:
            # TODO: when z3 gets unicode string support, case invariant matching
            # might need to be more complex. (see the casefold() builtin)
            return z3.Union(z3.Re(chr(arg).lower()), z3.Re(chr(arg).upper()))
        else:
            return z3.Re(chr(arg))
    elif op is RANGE:
        lo, hi = arg
        if re.IGNORECASE & flags:
            # TODO: when z3 gets unicode string support, case invariant matching
            # might need to be more complex. (see the casefold() builtin)
            return z3.Union(
                z3.Range(chr(lo).lower(), chr(hi).lower()),
                z3.Range(chr(lo).upper(), chr(hi).upper()),
            )
        else:
            return z3.Range(chr(lo), chr(hi))
    elif op is IN:
        return z3.Union(*(single_char_regex(a, flags) for a in arg))
    elif op is CATEGORY:
        if arg == CATEGORY_DIGIT:
            # TODO: when z3 gets unicode string support, we'll need to
            # extend this logic
            return z3.Range("0", "9")
        raise ReUnhandled
    elif op is ANY and arg is None:
        # TODO: when z3 gets unicode string support, we'll need to
        # revise this logic
        if re.DOTALL & flags:
            return z3.Range(z3.Unit(z3.BitVecVal(0, 8)), z3.Unit(z3.BitVecVal(255, 8)))
            # return z3.Range(chr(0), chr(127))
        else:
            return z3.Union(
                z3.Range(z3.Unit(z3.BitVecVal(0, 8)), z3.Unit(z3.BitVecVal(9, 8))),
                z3.Range(z3.Unit(z3.BitVecVal(11, 8)), z3.Unit(z3.BitVecVal(255, 8))),
            )
    else:
        return None


class _Match:
    def __init__(
        self, groups: List[Tuple[Optional[str], int, Union[int, SymbolicInt]]]
    ):  # (name, start, end)
        self._groups = groups
        self.lastindex = None
        self.lastgroup = None

    def __ch_realize__(self):
        self._groups = [
            (name, realize(start), realize(end)) for name, start, enf in self._groups
        ]

    def __bool__(self):
        return True

    def __repr__(self):
        return f"<re.Match object; span={self.span()!r}, match={self.group()!r}>"

    def __getitem__(self, idx):
        return self.group(idx)

    def _add_match(self, suffix_match):
        groups = [None] * max(len(self._groups), len(suffix_match._groups))
        for idx, g in enumerate(self._groups):
            groups[idx] = g
        for idx, g in enumerate(suffix_match._groups):
            if g is not None:
                groups[idx] = g
        (name, start, _) = self._groups[0]
        groups[0] = (name, start, suffix_match._groups[0][2])
        return _Match(groups)

    def _idx_for_group_name(self, group_name: str) -> int:
        for idx, triple in enumerate(self._groups):
            if triple[0] == group_name:
                return idx
        raise IndexError

    def group(self, *nums):
        if not nums:
            nums = (0,)
        ret = []
        for num in nums:
            if isinstance(num, str):
                num = self._idx_for_group_name(num)
            if self._groups[num] is None:
                ret.append(None)
            else:
                name, start, end = self._groups[num]
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
        ret = {}
        for groupname, start, end in self._groups:
            if groupname is not None:
                ret[groupname] = self.string[start:end]
        return ret

    def start(self, group=0):
        return self._groups[group][1]

    def end(self, group=0):
        return self._groups[group][2]

    def span(self, group=0):
        _, start, end = self._groups[group]
        return (start, end)


_REMOVE = object()


def _patt_replace(list_tree: List, from_obj: object, to_obj: object = _REMOVE):
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


def _slice_match_area(string, pos=0, endpos=None):
    smtstr = string.var
    if endpos is not None:
        smtstr = z3.SubString(smtstr, 0, endpos)
    return smtstr


_END_GROUP_MARKER = object()


def _internal_match_patterns(
    space: StateSpace, top_patterns: Any, flags: int, smtstr: z3.ExprRef, offset: int
) -> Optional[_Match]:
    """
    >>> from crosshair.statespace import SimpleStateSpace
    >>> import sre_parse
    >>> smtstr = z3.String('smtstr')
    >>> space = SimpleStateSpace()
    >>> space.add(smtstr == z3.StringVal('aabb'))
    >>> _internal_match_patterns(space, sre_parse.parse('a+'), 0, smtstr, 0).span()
    (0, 2)
    >>> _internal_match_patterns(space, sre_parse.parse('ab'), 0, smtstr, 1).span()
    (1, 3)
    """
    matchstr = z3.SubString(smtstr, offset, z3.Length(smtstr)) if offset > 0 else smtstr
    if len(top_patterns) == 0:
        return _Match([(None, offset, offset)])
    pattern = top_patterns[0]

    def continue_matching(prefix):
        suffix = _internal_match_patterns(
            space, top_patterns[1:], flags, smtstr, prefix.end()
        )
        if suffix is None:
            return None
        return prefix._add_match(suffix)

    # TODO: using a typed internal function triggers __hash__es inside the typing module.
    # Seems like this casues nondeterminism due to a global LRU cache used by the typing module.
    def fork_on(expr, sz):
        if space.smt_fork(expr):
            return continue_matching(_Match([(None, offset, offset + sz)]))
        else:
            return None

    # Handle simple single-character expressions using z3's built-in capabilities.
    z3_re = single_char_regex(pattern, flags)
    if z3_re is not None:
        ch = z3.SubString(matchstr, 0, 1)
        return fork_on(z3.InRe(ch, z3_re), 1)

    (op, arg) = pattern
    if op is MAX_REPEAT:
        (min_repeat, max_repeat, subpattern) = arg
        if max_repeat < min_repeat:
            return None
        reps = 0
        overall_match = _Match([(None, offset, offset)])
        while reps < min_repeat:
            submatch = _internal_match_patterns(
                space, subpattern, flags, smtstr, overall_match.end()
            )
            if submatch is None:
                return None
            overall_match = overall_match._add_match(submatch)
            reps += 1
        if max_repeat != MAXREPEAT and reps >= max_repeat:
            return continue_matching(overall_match)
        submatch = _internal_match_patterns(
            space, subpattern, flags, smtstr, overall_match.end()
        )
        if submatch is None:
            return continue_matching(overall_match)
        # we matched; try to be greedy first, and fall back to `submatch` as the last consumed match
        greedy_remainder = _patt_replace(
            top_patterns,
            arg,
            (
                1,
                max_repeat
                if max_repeat == MAXREPEAT
                else max_repeat - (min_repeat + 1),
                subpattern,
            ),
        )
        greedy_match = _internal_match_patterns(
            space, greedy_remainder, flags, smtstr, submatch.end()
        )
        if greedy_match is not None:
            return overall_match._add_match(submatch)._add_match(greedy_match)
        else:
            match_with_optional = continue_matching(overall_match._add_match(submatch))
            if match_with_optional is not None:
                return match_with_optional
            else:
                return continue_matching(overall_match)
    elif op is BRANCH and arg[0] is None:
        # NOTE: order matters - earlier branches are more greedily matched than later branches.
        branches = arg[1]
        first_path = list(branches[0]) + list(top_patterns)[1:]
        submatch = _internal_match_patterns(space, first_path, flags, smtstr, offset)
        # _patt_replace(top_patterns, pattern, branches[0])
        if submatch is not None:
            return submatch
        if len(branches) <= 1:
            return None
        else:
            return _internal_match_patterns(
                space,
                _patt_replace(top_patterns, branches, branches[1:]),
                flags,
                smtstr,
                offset,
            )
    elif op is AT:
        if arg in (AT_END, AT_END_STRING):
            if arg is AT_END and re.MULTILINE & flags:
                raise ReUnhandled("Multiline match with AT_END_STRING")
            return fork_on(matchstr == z3.StringVal(""), 0)
    elif op is SUBPATTERN:
        (groupnum, _a, _b, subpatterns) = arg
        if (_a, _b) != (0, 0):
            raise ReUnhandled("unsupported subpattern args")
        new_top = (
            list(subpatterns)
            + [(_END_GROUP_MARKER, (groupnum, offset))]
            + list(top_patterns)[1:]
        )
        return _internal_match_patterns(space, new_top, flags, smtstr, offset)
    elif op is _END_GROUP_MARKER:
        (group_num, begin) = arg
        match = continue_matching(_Match([(None, offset, offset)]))
        if match is None:
            return None
        while len(match._groups) <= group_num:
            match._groups.append(None)
        match._groups[group_num] = (None, begin, offset)
        return match
    raise ReUnhandled(op)


def _match_pattern(compiled_regex, pattern, orig_smtstr, pos, endpos=None):
    if pos == 0:
        # Remove some meaningless empty matchers for match/fullmatch:
        pattern = pattern.lstrip("^")
        while pattern.startswith(r"\A"):
            pattern = pattern[2:]

    space = orig_smtstr.statespace
    parsed_pattern = parse(pattern, compiled_regex.flags)
    smtstr = _slice_match_area(orig_smtstr, pos, endpos)
    match = _internal_match_patterns(
        space, parsed_pattern, compiled_regex.flags, smtstr, pos
    )
    if match is not None:
        match.pos = pos
        match.endpos = endpos if endpos is not None else len(orig_smtstr)
        match.re = compiled_regex
        match.string = orig_smtstr
        # fill None in unmatched groups:
        while len(match._groups) < compiled_regex.groups + 1:
            match._groups.append(None)
        # Link up any named groups:
        for name, num in compiled_regex.groupindex.items():
            (_, start, end) = match._groups[num]
            match._groups[num] = (name, start, end)
    return match


_orig_match = re.Pattern.match


def _match(self, string, pos=0, endpos=None):
    if type(string) is SymbolicStr:
        try:
            return _match_pattern(self, self.pattern, string, pos, endpos)
        except ReUnhandled as e:
            debug("Unable to symbolically analyze regular expression:", self.pattern, e)
    if endpos is None:
        return _orig_match(self, realize(string), pos)
    else:
        return _orig_match(self, realize(string), pos, endpos)


_orig_fullmatch = re.Pattern.fullmatch


def _fullmatch(self, string, pos=0, endpos=None):
    if type(string) is SymbolicStr:
        try:
            return _match_pattern(self, self.pattern + r"\Z", string, pos, endpos)
        except ReUnhandled as e:
            debug("Unable to symbolically analyze regular expression:", self.pattern, e)
    if endpos is None:
        return _orig_fullmatch(self, realize(string), pos)
    else:
        return _orig_fullmatch(self, realize(string), pos, endpos)


def make_registrations():
    register_patch(re.Pattern, with_realized_args(re.Pattern.search), "search")
    register_patch(re.Pattern, _match, "match")
    register_patch(re.Pattern, _fullmatch, "fullmatch")
    register_patch(re.Pattern, with_realized_args(re.Pattern.split), "split")
    register_patch(re.Pattern, with_realized_args(re.Pattern.findall), "findall")
    register_patch(re.Pattern, with_realized_args(re.Pattern.finditer), "finditer")
    register_patch(re.Pattern, with_realized_args(re.Pattern.sub), "sub")
    register_patch(re.Pattern, with_realized_args(re.Pattern.subn), "subn")
