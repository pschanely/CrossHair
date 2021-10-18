import doctest
import glob
import importlib
import os
import unittest

from crosshair import util

# Runs all the doctests in sibling *.py files


def load_tests(loader, tests, pattern):
    tests = unittest.TestSuite()
    root = os.path.split(__file__)[0]
    for path in glob.glob(os.path.join(root, "*.py")):
        (_, module_name) = util.extract_module_from_file(path)
        module = importlib.import_module(module_name)
        tests.addTests(doctest.DocTestSuite(module))
    return tests
