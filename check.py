import ast
import inspect
import importlib
import sys

import crosshairlib

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
    ret, report = crosshairlib.check_assertion_fn(fn_ast, fn_compiled, *a, **kw)
    if ret is True:
        msg = 'Proven by ' + ', '.join(strip_assert(s['name']) for s in report['statements'] if s['used'])
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
    

moduleinfo = crosshairlib.parse_pure(module)
for (name, fninfo) in moduleinfo.functions.items():
    print(' ===== ', name, ' ===== ')
    if name == '':
        for (assert_def, assert_compiled) in fninfo.get_assertions():
            scopes = crosshairlib.get_scopes_for_def(assert_def)
            check(assert_def, assert_compiled, scopes=scopes, src_loc=assert_def)
        continue
    fn = getattr(module, name)
    defining_assertions = fninfo.get_defining_assertions()
    # TODO: add an assertion that the function is a function
    # TODO: import the definitions of more things?
    assert_def = fninfo.definitional_assertion
    scopes = crosshairlib.get_scopes_for_def(fninfo.definition)
    check(assert_def, None, None, scopes=scopes, extra_support=defining_assertions, src_loc=fninfo.definition.returns)
    for (assert_def, assert_compiled) in fninfo.get_assertions():
        scopes = crosshairlib.get_scopes_for_def(assert_def)
        check(assert_def, assert_compiled, scopes=scopes, extra_support=defining_assertions, src_loc=assert_def)

print('info:{}:1:1:validation complete'.format(filename), file=sys.stderr)

sys.exit(1)
