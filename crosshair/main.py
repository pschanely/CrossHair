import argparse
import importlib
import importlib.util
import sys

from crosshair.core import *

def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CrossHair Analysis Tool')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--per_path_timeout', type=float)
    parser.add_argument('--per_condition_timeout', type=float)
    subparsers = parser.add_subparsers(help='sub-command help', dest='action')
    check_parser = subparsers.add_parser('check', help='Analyze one or more files')
    check_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    watch_parser = subparsers.add_parser('watch', help='Continuously watch and analyze files')
    watch_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    return parser
    
def process_level_options(command_line_args: argparse.Namespace) -> AnalysisOptions:
    options = AnalysisOptions()
    for optname in ('per_path_timeout', 'per_condition_timeout'):
        arg_val = getattr(command_line_args, optname)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options

def module_for_file(filepath:str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('crosshair.examples.tic_tac_toe', filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod) # type:ignore
    return mod

if __name__ == '__main__':
    any_errors = False
    args = command_line_parser().parse_args()
    set_debug(args.verbose)
    options = process_level_options(args)
    for name in args.files:
        if '=' in name:
            continue
        if name.endswith('.py'):
            _, name = extract_module_from_file(name)
        module = importlib.import_module(name)
        debug('Analyzing module ', module.__name__)
        for message in analyze_module(module, options):
            if message.state == MessageType.CANNOT_CONFIRM:
                continue
            desc = message.message
            if message.state == MessageType.POST_ERR:
                desc = 'Error while evaluating post condition: ' + desc
            debug(message.traceback)
            print('{}:{}:{}:{}:{}'.format('error', message.filename, message.line, message.column, desc))
            any_errors = True
    sys.exit(1 if any_errors else 0)
