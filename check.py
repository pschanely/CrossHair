import ast
import inspect
import importlib
import os
import sys

import crosshairlib

#sys.stdout = open(os.devnull, 'w')

filename = sys.argv[1]

def is_crosshair_file(filename, content):
    return os.path.split(filename)[1].endswith('crosshair.py') or content.startswith('from crosshair import *\n')

with open(filename) as fh:
    content = fh.read()
if not is_crosshair_file(filename, content):
    sys.exit(0)

modulename = inspect.getmodulename(filename)

module = importlib.import_module(modulename)

#from types import ModuleType
#module = ModuleType(modulename)
#module.__file__ = filename
#exec(compile(content, filename, 'exec'), module.__dict__)

any_errors = False

def report_message(severity, filename, line, col, message):
    print('{}:{}:{}:{}:{}'.format(severity, filename, line, col, message), file=sys.stderr)
    if severity == 'error':
        any_errors = True

def check(fn_ast, fn_compiled, *a, src_loc=None, **kw):
    try:
        ret, report = crosshairlib.check_assertion_fn(fn_ast, fn_compiled, *a, **kw)
    except crosshairlib.LocalizedError as e:
        report_message('error', filename, e.line, e.col+1, str(e))
        return
    except crosshairlib.ClientError as e:
        report_message('error', filename, src_loc.lineno, src_loc.col_offset, str(e))
        return
    if ret is True:
        msg = 'Proven by ' + ', '.join(sorted(s['name'] for s in report['statements'] if s['used']))
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
    

moduleinfo = crosshairlib.get_module_info(module)
print('', file=sys.stderr) # initial line appears to be ignored?
for (name, fninfo) in moduleinfo.functions.items():
    print(' ===== ', name, ' ===== ')
    #if name == '':
    #    for (assert_def, assert_compiled) in fninfo.get_assertions():
    #        scopes = crosshairlib.get_scopes_for_def(assert_def)
    #        check(assert_def, assert_compiled, scopes=scopes, src_loc=assert_def)
    #    continue
    fn = getattr(module, name)
    #scopes = crosshairlib.get_scopes_for_def(fninfo.definition)

    definition = fninfo.definition
    if crosshairlib.ch_option_true(definition, 'axiom', False):
        continue
    if definition.returns:
        defining_assertions = fninfo.get_defining_assertions()
        # indent the column offset a character, because otherwise emacs want to highlight the whitespace after the arrow:
        fake_returns_ast = ast.Num(0, lineno=fninfo.definition.returns.lineno, col_offset=fninfo.definition.returns.col_offset+1)
        check(fninfo.definition, None, None, extra_support=defining_assertions, src_loc=fake_returns_ast)
    #for (assert_def, assert_compiled) in fninfo.get_assertions():
    #    scopes = crosshairlib.get_scopes_for_def(assert_def)
    #    check(assert_def, assert_compiled, scopes=scopes, extra_support=defining_assertions, src_loc=assert_def)

#print('info:{}:1:1:validation complete'.format(filename), file=sys.stderr)

sys.exit(1 if any_errors else 0)
