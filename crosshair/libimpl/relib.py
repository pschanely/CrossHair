import re
from typing import *

from sre_parse import LITERAL, RANGE, ANY, IN, BRANCH, SUBPATTERN  # type: ignore
from sre_parse import MAX_REPEAT, MAXREPEAT  # type: ignore
from sre_parse import CATEGORY, CATEGORY_DIGIT  # type: ignore
from sre_parse import parse


import z3  # type: ignore

from crosshair import debug, register_patch, register_type
from crosshair import realize, with_realized_args, IgnoreAttempt

from crosshair.libimpl.builtinslib import SmtInt, SmtStr


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

def _handle_item(parsed: Tuple[object, Any], flags: int) -> z3.ExprRef:
    (op, arg) = parsed
    if op is LITERAL:
        if re.IGNORECASE & flags:
            if re.ASCII & flags:
                return z3.Union(z3.Re(chr(arg).lower()), z3.Re(chr(arg).upper()))
            else:
                raise ReUnhandled
        else:
            return z3.Re(chr(arg))
    elif op is RANGE:
        lo, hi = arg
        if re.IGNORECASE & flags:
            if re.ASCII & flags:
                return z3.Union(z3.Range(chr(lo).lower(), chr(hi).lower()),
                                z3.Range(chr(lo).upper(), chr(hi).upper()))
            else:
                raise ReUnhandled
        else:
            return z3.Range(chr(lo), chr(hi))
    elif op is IN:
        return z3.Union(*(_handle_item(a, flags) for a in arg))
    elif op is CATEGORY:
        if arg == CATEGORY_DIGIT:
            if re.ASCII & flags:
                return z3.Range('0','9')
        raise ReUnhandled
    elif op is ANY and arg is None:
        if re.ASCII & flags:
            if re.DOTALL & flags:
                return z3.Range(chr(0), chr(255))
            else:
                return z3.Union(z3.Range(chr(0), chr(9)),
                                z3.Range(chr(11), chr(255)))
        raise ReUnhandled
    elif op is BRANCH and arg[0] is None:
        branches = arg[1]
        return z3.Union(*(_handle_seq(b, flags) for b in branches))
    elif op is SUBPATTERN and arg[1] == 0 == arg[2]:
        group_num, _, _, subparsed = arg
        raise ReUnhandled  # need to figure out how to capture subpatterns
        #return _handle_seq(subparsed, flags)
    elif op is MAX_REPEAT:
        (min_repeat, max_repeat, subparsed) = arg
        if max_repeat == MAXREPEAT:
            if min_repeat == 0:
                return z3.Star(_handle_seq(subparsed, flags))
            elif min_repeat == 1:
                return z3.Plus(_handle_seq(subparsed, flags))
            else:
                raise ReUnhandled
        elif isinstance(min_repeat, int) and isinstance(max_repeat, int):
            return z3.Loop(_handle_seq(subparsed, flags), min_repeat, max_repeat)
        raise ReUnhandled
    else:
        raise ReUnhandled(str(op))

def _handle_seq(parsed: Any, flags: int) -> z3.ExprRef:
    if len(parsed) == 1:
        return _handle_item(parsed[0], flags)
    else:
        return z3.Concat(*(_handle_item(p, flags) for p in parsed))

def _interpret(pattern: str, flags: int):
    parsed = parse(pattern, flags)
    try:
        ret = _handle_seq(parsed, flags)
        debug('Attempting symbolic regex interpretation: ', ret)
        return ret
    except ReUnhandled:
        return None


class _Match:
    def __init__(self,
                 patt: re.Pattern,
                 string: str,
                 pos: int,
                 endpos: Optional[int],
                 groups: List[Tuple[Optional[str], int, int]]):
        self._groups = groups
        self.string = string
        self.pos = pos
        self.endpos = endpos if endpos is not None else len(string)
        self.re = patt
        self.lastindex = None
        self.lastgroup = None
    def __bool__(self):
        return True
    def __repr__(self):
        return f'<re.Match object; span={self.span()!r}, match={self.group()!r}>'
    def __getitem__(self, idx):
        return self.group(idx)
    def group(self, *nums):
        if not nums:
            nums = (0,)
        ret = []
        for num in nums:
            name, start, end = self._groups[num]
            ret.append(self.string[start:end])
        if len(nums) == 1:
            return ret[0]
        else:
            return tuple(ret)
    def groups(self):
        indicies = range(1, len(self._groups))
        if indicies:
            return self.group(*indicies)
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


def _slice_match_area(string, pos=0, endpos=None):
    smtstr = string.var
    is_bounded = pos != 0 or endpos is not None
    endpos = z3.Length(smtstr) if endpos is None else endpos
    if is_bounded:
        smtstr = z3.SubString(smtstr, pos, endpos - pos)
    return (smtstr, endpos)
    
_orig_match = re.Pattern.match
def _match(self, string, pos=0, endpos=None):
    # TODO: Work in progress. Greediness is not accounted for here.
    if type(string) is SmtStr:
        interp = _interpret(self.pattern, self.flags)
        if interp is not None:
            smtstr, endpos = _slice_match_area(string, pos, endpos)
            space = string.statespace
            match_end = SmtInt(space, int, 'matchend' + space.uniq())
            matching_substr = z3.SubString(smtstr, 0, match_end)
            if space.smt_fork(z3.InRe(matching_substr, interp)):
                ## It's the greediest match:
                #x = z3.Var(0, z3.IntSort())
                #space.add(z3.ForAll([x], z3.Implies(z3.And(match_end < x, x < z3.Length(smtstr))),
                #                    z3.Not(z3.InRe(z3.SubString(smtstr, 0, x), interp))))
                return _Match(self, string, pos, endpos, [(None, pos, match_end)])
            else:
                return None
        string = realize(string)
    return _orig_match(self, string, pos) if endpos is None else _orig_match(self, string, pos, endpos)

_orig_fullmatch = re.Pattern.fullmatch
def _fullmatch(self, string, pos=0, endpos=None):
    if type(string) is SmtStr:
        interp = _interpret(self.pattern, self.flags)
        if interp is not None:
            smtstr, endpos = _slice_match_area(string, pos, endpos)
            if string.statespace.smt_fork(z3.InRe(smtstr, interp)):
                return _Match(self, string, pos, endpos, [(None, pos, endpos)])
            else:
                return None
    return _orig_fullmatch(self, realize(string), self.flags)

def make_registrations():
    register_patch(re.Pattern, with_realized_args(re.Pattern.search), 'search')
    #register_patch(re.Pattern, with_realized_args(re.Pattern.match), 'match')
    #register_patch(re.Pattern, _match, 'match')
    register_patch(re.Pattern, _fullmatch, 'fullmatch')
    register_patch(re.Pattern, with_realized_args(re.Pattern.split), 'split')
    register_patch(re.Pattern, with_realized_args(re.Pattern.findall), 'findall')
    register_patch(re.Pattern, with_realized_args(re.Pattern.finditer), 'finditer')
    register_patch(re.Pattern, with_realized_args(re.Pattern.sub), 'sub')
    register_patch(re.Pattern, with_realized_args(re.Pattern.subn), 'subn')
