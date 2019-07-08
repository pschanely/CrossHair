import ast
import importlib
import inspect
import os.path
from typing import Dict, List, Optional, Callable, Tuple, Any
from crosshair.util import extract_module_from_file
import mypy
import mypy.options
import mypy.build
from mypy.fscache import FileSystemCache
from mypy.nodes import Node, MypyFile
from mypy.find_sources import create_source_list
from mypy.traverser import TraverserVisitor
import types

class NameResolver(TraverserVisitor):
    def __init__(self):
        self.byline = {}
    def visit_name_expr(self, node: mypy.nodes.NameExpr) -> None:
        if hasattr(node, 'node'):
            self.byline[(node.line, node.column)] = node.node
        
def resolve_names(mypynode:Node) -> Dict[Tuple[int, int], Node]:
    resolver = NameResolver()
    mypynode.accept(resolver)
    return resolver.byline

class AstAnnotator(ast.NodeVisitor):
    def __init__(self, annotations):
        self.annotations = annotations
    def visit(self, node):
        key = (node.lineno, node.col_offset)
        mypynode = self.annotations.get(key)
        if mypynode:
            node.node = mypynode
        self.generic_visit(node)

def annotate_ast(astnode, mypynode):
    annotator = AstAnnotator(resolve_names(mypynode))
    annotator.visit(astnode)

class MyPyInfo:
    modules: Dict[str, MypyFile]
    asts: Dict[str, ast.Module]

    def __init__(self, mgr, asts):
        self.modules = mgr.modules
        self.asts = asts

    def load_module(self, mod) -> Optional[ast.Module]:
        filename = inspect.getsourcefile(mod)
        fileast = self.asts[filename]
        mypymodule = self.modules[mod.__name__]
        annotate_ast(fileast, mypymodule)
        return fileast
        
    def lookup(self, fn: Callable) -> Optional[ast.FunctionDef]:
        mod = importlib.import_module(fn.__module__)
        modast = self.load_module(mod)
        if not modast:
            return None
        for definition in modast.body:
            if not isinstance(definition, ast.FunctionDef):
                continue
            if definition.name == fn.__name__:
                return definition
        return None
    

def mypy_info(filenames: List[str]) -> MyPyInfo:
    opts = mypy.options.Options()
    
    os.putenv('PYTHONPATH', os.path.pathsep.join(set(extract_module_from_file(f)[0] for f in filenames)))
    print(' === ',filenames, os.getenv('PYTHONPATH'))

    file_to_ast = {}
    for filename in filenames:
        with open(filename, 'r') as fh:
            file_to_ast[filename] = ast.parse(fh.read())
    
    fscache = FileSystemCache()

    # One error-free build causes mypy to not check the file again,
    # causing us to get an empty build. Turn off caching until we
    # find a better way to force out the cache
    # Update: IDK if above is true. Things are still slow though when "on".
    opts.mypyc = False
    opts.incremental = True
    opts.fine_grained_incremental = True
    opts.cache_fine_grained = True
    opts.use_fine_grained_cache = True
    opts.show_traceback = True

    def flusher(lines: List[str], _: bool) -> None:
        for line in lines:
            print(line)
    targets = create_source_list(filenames, opts, fscache)
    mgr = mypy.build.build(targets, opts, flush_errors=flusher).manager
    return MyPyInfo(mgr, file_to_ast)

# Some functions that are used by the tests

def _useless1(t: Tuple[int, ...]) -> Any: #Tuple[int, ...]:
    return t + t

def _useless2():
    import mypy.options as ox
    O = ox.Options
    opts = O()
    return opts
