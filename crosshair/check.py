import argparse
import hashlib
import importlib
import inspect
import os
import sys
import time
import traceback
import types
import z3  # type: ignore

from typing import *
from crosshair.util import debug
from crosshair.base import Condition, get_conditions, CheckStatus
from crosshair.analysis_server import find_or_spawn_server


ANY_ERRORS = False


def report_message(severity: str, filename: str,
                   line: int, col: int, message: str) -> None:
    global ANY_ERRORS
    print('{}:{}:{}:{}:{}'.format(severity, filename, line, col, message))
    if severity == 'error':
        ANY_ERRORS = True


def full_module_name_for_file(filename: str) -> str:
    dirs = [m for m in [inspect.getmodulename(filename)] if m]
    path = os.path.split(filename)[0]
    while os.path.exists(os.path.join(path, '__init__.py')):
        path, cur = os.path.split(path)
        dirs.append(cur)
    dirs.reverse()
    return '.'.join(dirs)


def extract_conditions(
        container: Union[types.ModuleType, Type],
        containing_module: types.ModuleType) -> List[Tuple[str, Condition]]:
    fns: List[Tuple[str, Condition]] = []
    for (name, obj) in inspect.getmembers(container):
        if inspect.getmodule(obj) != containing_module:
            continue
        if inspect.isfunction(obj):
            conditions = get_conditions(obj)
            for condition in conditions:
                if condition.z3expr is not None:
                    debug('z3 expr :', condition.z3expr)
                    solver = z3.Solver()
                    solver.add(condition.z3expr)
                    check_result = str(solver.check())
                    debug('z3 solver result : ', repr(check_result))
                    filename, line_num = condition.src_info
                    if check_result == 'sat':
                        msg = ('Proven false by SMT solver; '
                               'counterexample: ') + str(solver.model())
                        report_message('error', filename, line_num, 0, msg)
                        # do not bother with assertions that SMT can discharge:
                        continue
                    elif check_result == 'unsat':
                        report_message('info', filename, line_num, 0,
                                       'Proven true by SMT solver')
                        # do not bother with assertions that SMT can discharge:
                        continue
                    debug('z3 solver not helpful')
                fns.append((name, condition))
        elif inspect.isclass(obj):
            classname = obj.__name__
            fns.extend([(classname + '.' + name, condition) for name, condition
                        in extract_conditions(obj, containing_module)])
    return fns


def secs_since_new_path(plot_data: List[Dict[str, str]]) -> Optional[float]:
    if (not plot_data) or len(plot_data) < 2:
        return None
    ct = int(plot_data[-1]['paths_total'])
    for entry in reversed(plot_data):
        if int(entry['paths_total']) < ct:
            return time.time() - float(entry['unix_time'])
    return None


def main(filename: str, opts: Mapping[str, object]) -> None:
    if not os.path.exists(filename):
        raise FileNotFoundError()
    with open(filename, 'rb') as fh:
        content_hash = hashlib.sha224(fh.read()).hexdigest()
    modulename = full_module_name_for_file(filename)
    # debug('module name ', modulename)
    try:
        module = importlib.import_module(modulename)
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            raise e
        if hasattr(e, 'offset'):
            report_message('error', filename, getattr(e, 'lineno', 0),
                           getattr(e, 'offset', 0), str(e))
        else:
            tb = sys.exc_info()[2]
            (filename, lineno, fn_name, text) = traceback.extract_tb(tb)[-1]
            report_message('error', filename, lineno, 0, str(e))
        sys.exit(1)
    # debug(module)
    fns = extract_conditions(module, module)
    if not fns:
        return

    server = find_or_spawn_server()
    debug('Using server at ', server.homedir)
    cmds = server.get_commands()
    cmds['target'] = {'package': module.__package__,
                      'module': module.__name__,
                      'fns': sorted(set(name for (name, c) in fns))}
    debug('target', cmds['target'])
    cmds['target_content_hash'] = content_hash

    server.write_commands(cmds)
    status = server.get_status()
    plotdata = status['plotdata']
    examples = status['examples']
    examples.add(bytes(b''))
    debug('examples', examples)
    examples = list(map(bytearray, examples))

    for _fn_name, condition in fns:
        filename, line_num = condition.src_info

        statuses = list((example, *condition.check_buffer(example))
                        for example in examples)
        # We want the smallest example in the worst severity:
        statuses.sort(key=lambda s: (s[1], -len(s[0])))
        if opts.get('dump_examples'):
            for s in statuses:
                print(s[1], condition.post,
                      condition.format_call(*condition.unpack_args(s[0])),
                      *s[2:])
        if statuses[-1][1].is_failure():
            print('smallest of worst: ', statuses[-1])
            args, kwargs = condition.unpack_args(statuses[-1][0])
            args, kwargs = condition.simplify_args(args, kwargs)
            msg = condition.fails_on_args(args, kwargs)
            assert msg is not None
            report_message('error', filename, line_num, 0, msg)
        elif any(s == CheckStatus.Ok for _, s, _, _ in statuses):
            secs = secs_since_new_path(plotdata)
            if secs is None:
                msg = 'Looks good so far (restarting search now)'
                report_message('info', filename, line_num, 0, msg)
            else:
                num_paths = plotdata[-1]['paths_total']
                msg = ('Looks good so far (' + num_paths + ' paths found; ' +
                       str(round(secs / 60, 1)) + ' min since last discovery)')
                report_message('info', filename, line_num, 0, msg)
        elif len(examples) > 1:
            args, kwargs = condition.unpack_args(statuses[-1][0])
            msg = ('Having trouble meeting preconditions. ' +
                   statuses[-1][1].name + ' when calling ' +
                   condition.format_call(args, kwargs))
            report_message('warning', filename, line_num, 0, msg)
        else:
            report_message('warning', filename, line_num, 0, 'Checking...')

    global ANY_ERRORS
    sys.exit(1 if ANY_ERRORS else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check crosshair invariants.')
    parser.add_argument(
        'file', metavar='F', type=str, help='the file to check')
    parser.add_argument(
        '--dump-examples', action='store_true',
        help='print all the example inputs used for evaluation')
    args = parser.parse_args()
    filename = args.file
    try:
        main(filename, args.__dict__)
    except Exception as e:
        report_message('error', filename, 0, 0,
                       'Crosshair did not check: ' + str(e))
        raise
