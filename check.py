import ast
import inspect
import importlib
import json
import os
import sys
import traceback

import dirdict
import crosshairlib

#sys.stdout = open(os.devnull, 'w')

filename = sys.argv[1]

proofcache = dirdict.DirDict(os.path.abspath('.crosshair_proof_cache'))


def is_crosshair_file(filename, content):
    return os.path.split(filename)[1].endswith('crosshair.py') or content.startswith('from crosshair import *\n')

with open(filename) as fh:
    content = fh.read()
if not is_crosshair_file(filename, content):
    sys.exit(0)

any_errors = False

def report_message(severity, filename, line, col, message):
    global any_errors
    print('{}:{}:{}:{}:{}'.format(severity, filename, line, col, message), file=sys.stderr)
    if severity == 'error' or severity == 'warning':
        any_errors = True

modulename = inspect.getmodulename(filename)

module = None
try:
    module = importlib.import_module(modulename)
except Exception as e:
    if hasattr(e, 'offset'):
        report_message('error', filename, getattr(e, 'lineno', 0), getattr(e, 'offset', 0), str(e))
    else:
        tb = sys.exc_info()[2]
        (filename, lineno, fn_name, text) = traceback.extract_tb(tb)[-1]
        report_message('error', filename, lineno, 0, str(e))
    sys.exit(1)

#from types import ModuleType
#module = ModuleType(modulename)
#module.__file__ = filename
#exec(compile(content, filename, 'exec'), module.__dict__)

def check(fn_ast, fn_compiled, *a, src_loc=None, **kw):
    try:
        options = {'proof_cache': proofcache}
        ret, report = crosshairlib.check_assertion_fn(fn_ast, fn_compiled, *a, options=options, **kw)
    except crosshairlib.LocalizedError as e:
        report_message('error', e.filename, e.line, e.col+1, str(e))
        return
    except crosshairlib.ClientError as e:
        report_message('error', filename, src_loc.lineno, src_loc.col_offset, str(e))
        return
    if ret is True:
        msg = 'Proven by ' + ', '.join(sorted("'"+s['name']+"'" for s in report['statements'] if s['used']))
        severity = 'info'
    elif ret is False:
        msg = 'untrue'
        severity = 'error'
        if 'counterexample' in report:
            counterexample = report['counterexample']
            msg = 'not when ' + ' and '.join(str(k)+'='+repr(counterexample[k]) for k in counterexample)
    else:
        msg = 'unable to determine'
        severity = 'warning'
    if src_loc is None:
        src_loc = fn_ast
    report_message(severity, filename, src_loc.lineno, src_loc.col_offset, msg)

def main():
    try:
        moduleinfo = crosshairlib.get_module_info(module)
    except crosshairlib.LocalizedError as e:
        report_message('error', e.filename, e.line, e.col+1, str(e))
        sys.exit(1)

    print('', file=sys.stderr) # initial line appears to be ignored?
    for (name, fninfo) in moduleinfo.functions.items():
        print(' ===== ', name, ' ===== ')
        fn = getattr(module, name)

        definition = fninfo.definition
        if crosshairlib.ch_option_true(definition, 'axiom', False):
            # TODO: error on things with ch annotations that are missing return type declarations?
            continue
        if definition.returns:
            defining_assertions = fninfo.get_defining_assertions()
            # indent the column offset a character, because otherwise emacs want to highlight the whitespace after the arrow:
            fake_returns_ast = ast.Num(0, lineno=fninfo.definition.returns.lineno, col_offset=fninfo.definition.returns.col_offset+1)
            check(fninfo.definition, None, None, extra_support=defining_assertions, src_loc=fake_returns_ast)

import cProfile
cProfile.run('main()', 'prof.prof')

#sys.exit(1 if any_errors else 0)
