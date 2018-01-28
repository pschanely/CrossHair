import ast
import inspect
import importlib
import os
import sys

import crosshairlib

#sys.stdout = open(os.devnull, 'w')

filename = sys.argv[1]

with open(filename) as fh:
    content = fh.read()
if not content.startswith('from crosshair import *\n'):
    sys.exit(0)

modulename = inspect.getmodulename(filename)

module = importlib.import_module(modulename)

#from types import ModuleType
#module = ModuleType(modulename)
#module.__file__ = filename
#exec(compile(content, filename, 'exec'), module.__dict__)

def strip_assert(name):
    if name.startswith('_assert_'):
        return name[len('_assert_'):]
    return name

def check(fn_ast, fn_compiled, *a, src_loc=None, **kw):
    try:
        ret, report = crosshairlib.check_assertion_fn(fn_ast, fn_compiled, *a, **kw)
    except crosshairlib.ResolutionError as e:
        print('{}:{}:{}:{}:{}'.format('error', filename, e.line, e.col+1, 'Undefined: "'+e.identifier+'"'), file=sys.stderr)
        return
    if ret is True:
        msg = 'Proven by ' + ', '.join(sorted(strip_assert(s['name']) for s in report['statements'] if s['used']))
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
    print('{}:{}:{}:{}:{}'.format(severity, filename, src_loc.lineno, src_loc.col_offset, msg), file=sys.stderr)
    #sys.stderr.flush() # flushing doesn't seem to matter  :(
    

moduleinfo = crosshairlib.get_module_info(module)
print('', file=sys.stderr) # initial line appears to be ignored?
for (name, fninfo) in moduleinfo.functions.items():
    print(' ===== ', name, ' ===== ')
    if name == '':
        for (assert_def, assert_compiled) in fninfo.get_assertions():
            scopes = crosshairlib.get_scopes_for_def(assert_def)
            check(assert_def, assert_compiled, scopes=scopes, src_loc=assert_def)
        continue
    fn = getattr(module, name)
    defining_assertions = fninfo.get_defining_assertions()
    scopes = crosshairlib.get_scopes_for_def(fninfo.definition)

    # indent the column offset a character, because otherwise emacs want to highlight the whitespace after the arrow:
    if fninfo.definition.returns:
        fake_returns_ast = ast.Num(0, lineno=fninfo.definition.returns.lineno, col_offset=fninfo.definition.returns.col_offset+1)
        check(fninfo.definitional_assertion, None, None, scopes=scopes, extra_support=defining_assertions, src_loc=fake_returns_ast)
    for (assert_def, assert_compiled) in fninfo.get_assertions():
        scopes = crosshairlib.get_scopes_for_def(assert_def)
        check(assert_def, assert_compiled, scopes=scopes, extra_support=defining_assertions, src_loc=assert_def)

#print('info:{}:1:1:validation complete'.format(filename), file=sys.stderr)

sys.exit(1)
