import shutil
import sys
import tempfile
import io
import unittest
from argparse import Namespace
from os.path import join
from typing import *

from crosshair.core_and_libs import AnalysisOptions
from crosshair.util import add_to_pypath, NotFound

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
    retcode = check(Namespace(files=files), options, buf)
    lines = [l for l in buf.getvalue().split('\n') if l]
    return retcode, lines

SIMPLE_FOO = {
            'foo.py': """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return x + 1
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


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
