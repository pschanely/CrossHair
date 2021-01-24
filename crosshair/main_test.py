import shutil
import sys
import tempfile
import io
import unittest
from argparse import Namespace
from os.path import join
from typing import *

from crosshair.core_and_libs import AnalysisOptions
from crosshair.fnutil import NotFound
from crosshair.util import add_to_pypath

from crosshair.main import *


def simplefs(path: str, files:dict) -> None:
    for name, contents in files.items():
        subpath = join(path, name)
        if isinstance(contents, str):
            with open(subpath, 'w') as fh:
                fh.write(contents)
        elif isinstance(contents, dict):
            os.mkdir(subpath)
            simplefs(subpath, contents)
        else:
            raise Exception('bad input to simplefs')

def call_check(files: List[str], options=None) -> Tuple[int, List[str]]:
    if options is None:
        options = AnalysisOptions()
    buf: io.StringIO = io.StringIO()
    retcode = check(Namespace(file=files), options, buf)
    lines = [l for l in buf.getvalue().split('\n') if l]
    return retcode, lines

def call_diffbehavior(fn1: str, fn2: str, options=None) -> Tuple[int, List[str]]:
    if options is None:
        options = AnalysisOptions()
    buf: io.StringIO = io.StringIO()
    errbuf: io.StringIO = io.StringIO()
    retcode = diffbehavior(Namespace(fn1=fn1, fn2=fn2), options, buf, errbuf)
    lines = [l for l in buf.getvalue().split('\n')+errbuf.getvalue().split('\n') if l]
    return retcode, lines

SIMPLE_FOO = {
            'foo.py': """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return x + 1
"""
}

ASSERT_BASED_FOO = {
            'foo.py': """
def foofn(x: int) -> int:
  assert x >= 100
  x = x + 1
  assert x != 101
  return x
"""
}

FOO_WITH_CONFIRMABLE_AND_PRE_UNSAT = {
            'foo.py': """
def foo_confirmable(x: int) -> int:
  ''' post: _ > x '''
  return x + 1
def foo_pre_unsat(x: int) -> int:
  '''
  pre: x != x
  post: True
  '''
  return x
"""
}

OUTER_INNER = {
            'outer': {
                '__init__.py': '',
                'inner.py': """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return x
"""
            }
}

CIRCULAR_WITH_GUARD = {
    'first.py': """
import typing
if typing.TYPE_CHECKING:
    from second import Second
class First():
    def __init__(self, f: "Second") -> None:
        ''' post: True '''
""",
    'second.py': """
from first import First
class Second():
    pass
"""
    }

class MainTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.orig_modules = sys.modules.copy()

    def tearDown(self):
        shutil.rmtree(self.root)
        sys.modules = self.orig_modules

    def test_load_file(self):
        simplefs(self.root, SIMPLE_FOO)
        module = load_file(join(self.root, 'foo.py'))
        self.assertNotEqual(module, None)
        self.assertEqual(module.foofn(5), 6)

    def test_check_by_filename(self):
        simplefs(self.root, SIMPLE_FOO)
        retcode, lines = call_check([join(self.root, 'foo.py')])
        self.assertEqual(retcode, 2)
        self.assertEqual(len(lines), 1)
        self.assertIn('foo.py:3:error:false when calling foofn', lines[0])

    def test_check_via_main(self):
        simplefs(self.root, SIMPLE_FOO)
        try:
            sys.stdout = io.StringIO()
            with self.assertRaises(SystemExit) as ctx:
                main(['check', join(self.root, 'foo.py')])
            self.assertEqual(2, ctx.exception.code)
        finally:
            sys.stdout = sys.__stdout__

    def test_assert_mode_e2e(self):
        simplefs(self.root, ASSERT_BASED_FOO)
        try:
            sys.stdout = io.StringIO()
            with self.assertRaises(SystemExit) as ctx:
                main(['check', join(self.root, 'foo.py'), '--analysis_kind=asserts'])
        finally:
            out = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
        self.assertEqual(ctx.exception.code, 2)
        # TODO: check filename and line number (these are wrong currently)
        self.assertRegex(
            out, r'foo.py\:5\:error\:AssertionError\:  when calling foofn\(x \= 100\)')
        self.assertEqual(len([l for l in out.split('\n') if l]), 1)

    def test_report_confirmation(self):
        simplefs(self.root, FOO_WITH_CONFIRMABLE_AND_PRE_UNSAT)
        retcode, lines = call_check([join(self.root, 'foo.py')])
        self.assertEqual(retcode, 0)
        self.assertEqual(lines, [])
        # Now, turn on confirmations with the `--report_all` option:
        retcode, lines = call_check([join(self.root, 'foo.py')],
                                    options=AnalysisOptions(report_all=True))
        self.assertEqual(retcode, 0)
        self.assertEqual(len(lines), 2)
        output_text = '\n'.join(lines)
        self.assertIn('foo.py:3:info:Confirmed over all paths.', output_text)
        self.assertIn('foo.py:7:info:Unable to meet precondition.', output_text)

    def test_check_nonexistent_filename(self):
        simplefs(self.root, SIMPLE_FOO)
        retcode, lines = call_check([join(self.root, 'notexisting.py')])
        self.assertEqual(retcode, 2)
        self.assertEqual(len(lines), 1)
        self.assertIn("error:No module named 'notexisting'", lines[0])

    def test_check_by_module(self):
        simplefs(self.root, SIMPLE_FOO)
        with add_to_pypath(self.root):
            retcode, lines = call_check(['foo'])
            self.assertEqual(retcode, 2)
            self.assertEqual(len(lines), 1)
            self.assertIn('foo.py:3:error:false when calling foofn', lines[0])

    def test_check_nonexistent_module(self):
        simplefs(self.root, SIMPLE_FOO)
        retcode, lines = call_check(['notexisting'])
        self.assertEqual(retcode, 2)
        self.assertEqual(len(lines), 1)
        self.assertIn("error:No module named 'notexisting'", lines[0])

    def test_check_by_package(self):
        simplefs(self.root, OUTER_INNER)
        with add_to_pypath(self.root):
            retcode, lines = call_check(['outer.inner.foofn'])
            self.assertEqual(retcode, 0)
            self.assertEqual(len(lines), 0)

    def test_check_nonexistent_member(self):
        simplefs(self.root, OUTER_INNER)
        with add_to_pypath(self.root):
            self.assertRaises(NotFound, lambda: call_check(['outer.inner.nonexistent']))

    def test_check_circular_with_guard(self):
        simplefs(self.root, CIRCULAR_WITH_GUARD)
        with add_to_pypath(self.root):
            retcode, lines = call_check([join(self.root, 'first.py')])
            self.assertEqual(retcode, 0)

    def test_diff_behavior_same(self):
        simplefs(self.root, SIMPLE_FOO)
        with add_to_pypath(self.root):
            retcode, lines = call_diffbehavior('foo.foofn', 'foo.foofn')
            self.assertEqual(retcode, 0)
            self.assertEqual(lines, [
                'No differences found. (attempted 2 iterations)',
                'All paths exhausted, functions are likely the same!'
            ])

    def test_diff_behavior_different(self):
        simplefs(self.root, {'foo.py': """
def add(x: int, y: int) -> int:
  return x + y
def faultyadd(x: int, y: int) -> int:
  return 42 if (x, y) == (10, 10) else x + y
"""})
        with add_to_pypath(self.root):
            retcode, lines = call_diffbehavior('foo.add', 'foo.faultyadd')
            self.assertEqual(retcode, 1)
            self.assertEqual(
                lines,
                ['Given: (x=10, y=10),',
                 '        foo.add : returns 20',
                 '  foo.faultyadd : returns 42'])

    def test_diff_behavior_error(self):
        retcode, lines = call_diffbehavior('foo.unknown', 'foo.unknown')
        self.assertEqual(retcode, 2)
        retcode, lines = call_diffbehavior('foo.unknown', 'foo.unknown')
        self.assertRegex(lines[0], '.*ModuleNotFoundError')

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
