import re
import sre_parse
import sys
import unittest
from typing import *

import z3  # type: ignore

from crosshair.libimpl.relib import _handle_seq

from crosshair.core_and_libs import *
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_post_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_unknown
from crosshair.test_util import check_messages
from crosshair.util import set_debug

class RegularExpressionTests(unittest.TestCase):

    def test_handle_simple(self):
        z3re = _handle_seq(sre_parse.parse('abc'), 0)
        self.assertEqual(z3.simplify(z3.InRe('abc', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('ab', z3re)), False)

    def test_handle_or(self):
        z3re = _handle_seq(sre_parse.parse('a|bc'), 0)
        self.assertEqual(z3.simplify(z3.InRe('bc', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('a', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('ac', z3re)), False)

    def test_handle_noncapturing_subgroup(self):
        z3re = _handle_seq(sre_parse.parse('(?:a|b)c'), 0)
        self.assertEqual(z3.simplify(z3.InRe('ac', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('bc', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('a', z3re)), False)

    def test_handle_range(self):
        z3re = _handle_seq(sre_parse.parse('[a-z]7'), 0)
        self.assertEqual(z3.simplify(z3.InRe('b7', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('z7', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('A7', z3re)), False)

    def test_handle_ascii_wildcard(self):
        z3re = _handle_seq(sre_parse.parse('1.2'), re.A)
        self.assertEqual(z3.simplify(z3.InRe('1x2', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('1\x002', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('111', z3re)), False)

    def test_handle_repeats(self):
        z3re = _handle_seq(sre_parse.parse('y*e+s{2,3}'), 0)
        self.assertEqual(z3.simplify(z3.InRe('yess', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('ess', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('yyesss', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('yss', z3re)), False)
        self.assertEqual(z3.simplify(z3.InRe('yessss', z3re)), False)
        self.assertEqual(z3.simplify(z3.InRe('e', z3re)), False)

    def test_handle_ascii_numeric(self):
        z3re = _handle_seq(sre_parse.parse('a\d+'), re.A)
        self.assertEqual(z3.simplify(z3.InRe('a32', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('a0', z3re)), True)
        self.assertEqual(z3.simplify(z3.InRe('a-', z3re)), False)

    def test_fullmatch_basic_fail(self) -> None:
        def f(s: str) -> bool:
            ''' post: _ '''
            return not re.compile('ab+').fullmatch(s)
        self.assertEqual(*check_fail(f))

    def test_fullmatch_basic_ok(self) -> None:
        def f(s: str) -> Optional[re.Match]:
            '''
            pre: s == 'a'
            post: _
            '''
            return re.compile('a').fullmatch(s)
        self.assertEqual(*check_ok(f))

    def test_fullmatch_complex_fail(self) -> None:
        def f(s: str) -> str:
            '''
            pre: re.fullmatch('ab+aXb+a+', s)
            post: _ != 'X'
            '''
            return s[-5]
        self.assertEqual(*check_fail(f))

    def TODO_test_match_basic_fail(self) -> None:
        def f(s: str) -> bool:
            ''' post: implies(_, len(s) <= 3) '''
            return re.compile('ab?c').match(s)
        self.assertEqual(*check_ok(f))
        
    def test_match_properties(self) -> None:
        test_string = '01ab9'
        match = re.compile('ab').fullmatch('01ab9', 2, 4)
        assert match is not None
        self.assertEqual(match.span(), (2, 4))
        self.assertEqual(match.groups(), ())
        self.assertEqual(match.group(0), 'ab')
        self.assertEqual(match[0], 'ab')
        self.assertEqual(match.pos, 2)
        self.assertEqual(match.endpos, 4)
        self.assertEqual(match.lastgroup, None)
        self.assertEqual(match.string, '01ab9')
        self.assertEqual(match.re.pattern, 'ab')
        def f(s:str) -> Optional[re.Match]:
            '''
            pre: s == '01ab9'
            post: _.span() == (2, 4)
            post: _.groups() == ()
            post: _.group(0) == 'ab'
            post: _[0] == 'ab'
            post: _.pos == 2
            post: _.endpos == 4
            post: _.lastgroup == None
            post: _.string == '01ab9'
            post: _.re.pattern == 'ab'
            '''
            return re.compile('ab').fullmatch(s, 2, 4)
        self.assertEqual(*check_ok(f))


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
