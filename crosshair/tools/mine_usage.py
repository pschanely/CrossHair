"""
Mine a *usage prior* for the support corpus: how often each operation is used in
real code.  Static (AST) call-site counting -- two metrics per operation:

  * ``sites``    -- total call sites / operator occurrences across the corpus
  * ``packages`` -- number of DISTINCT packages that use it (breadth; robust to
                    one big repetitive package dominating)

The treemap's cells are TYPED ops but source text is not.  Attribution by class:
  * name-unambiguous (builtin ``len``, qualified ``math.sqrt``, ``from x import
    y``) -> attributed exactly.
  * type-ambiguous (operators ``a + b``, bare methods ``x.replace(...)``) -> we
    try to RESOLVE the receiver/operand type from local evidence (literal
    operands like ``"x" + s`` or ``b"..".replace``; vars with a known type from
    annotations or literal assignments).  Resolved uses are attributed directly;
    the unresolved remainder is apportioned by the type distribution observed
    among the resolved ones (even split only when there's no evidence at all).

Cell-key schema comes from a measured JSON (so weights join onto treemap cells):

    python -m crosshair.tools.mine_usage --measured surface.json,funcs.json \\
        --corpus /path/to/site-packages --out usage.json
"""

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

_Index = Tuple[
    Set[str],
    DefaultDict[str, Set[str]],
    Set[str],
    DefaultDict[str, Set[str]],
]

_BIN = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitAnd: "__and__",
    ast.BitXor: "__xor__",
    ast.MatMult: "__matmul__",
}
_UNARY = {ast.USub: "__neg__", ast.UAdd: "__pos__", ast.Invert: "__invert__"}
_CMP = {
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.In: "__contains__",
    ast.NotIn: "__contains__",
}
_BUILTIN_TYPES = {
    "str",
    "bytes",
    "bytearray",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "int",
    "float",
    "bool",
}


def build_index(measured: Dict[str, Any]) -> _Index:
    """measured cell-keys -> (builtin_funcs, stdlib_funcs{mod:set}, modules, method_keys{meth:set})."""
    builtin_funcs, stdlib_funcs, modules, method_keys = (
        set(),
        defaultdict(set),
        set(),
        defaultdict(set),
    )
    for k in measured:
        if k.startswith("builtins.") and k.endswith("_method"):
            _typ, meth = k[len("builtins.") : -len("_method")].split("_", 1)
            method_keys[meth].add(k)
        elif k.startswith("builtins."):
            builtin_funcs.add(k[len("builtins.") :])
        elif "." in k:
            mod, fn = k.rsplit(".", 1)
            modules.add(mod)
            stdlib_funcs[mod].add(fn)
    return builtin_funcs, stdlib_funcs, modules, method_keys


def _const_type(node: ast.AST) -> Optional[str]:
    """builtin type name for a literal / collection / constructor node, else None."""
    if isinstance(node, ast.Constant):
        t = type(node.value).__name__
        return t if t in _BUILTIN_TYPES else None
    if isinstance(node, (ast.JoinedStr,)):
        return "str"
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _BUILTIN_TYPES
    ):
        return node.func.id
    return None


def _ann_type(node: ast.AST) -> Optional[str]:
    """base builtin type name for a type annotation, else None."""
    if isinstance(node, ast.Name):
        return node.id if node.id in _BUILTIN_TYPES else None
    if isinstance(node, ast.Subscript):
        base = (
            node.value.id
            if isinstance(node.value, ast.Name)
            else getattr(node.value, "attr", None)
        )
        return {
            "List": "list",
            "Dict": "dict",
            "Set": "set",
            "Tuple": "tuple",
            "FrozenSet": "frozenset",
        }.get(base, base if base in _BUILTIN_TYPES else None)
    return None


def _var_types(tree: ast.Module) -> Dict[str, str]:
    """file-level name -> builtin type from annotations and literal assignments
    (conflicting names dropped)."""
    t, bad = {}, set()

    def put(name: str, typ: Optional[str]) -> None:
        if not typ or name in bad:
            return
        if name in t and t[name] != typ:
            t.pop(name)
            bad.add(name)
        else:
            t[name] = typ

    for n in ast.walk(tree):
        if isinstance(n, ast.arg) and n.annotation:
            put(n.arg, _ann_type(n.annotation))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            put(n.target.id, _ann_type(n.annotation))
        elif isinstance(n, ast.Assign):
            ct = _const_type(n.value)
            if ct:
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name):
                        put(tgt.id, ct)
    return t


def _expr_type(node: ast.AST, vt: Dict[str, str]) -> Optional[str]:
    return _const_type(node) or (
        vt.get(node.id) if isinstance(node, ast.Name) else None
    )


def file_tokens(
    tree: ast.Module, idx: _Index, vt: Dict[str, str]
) -> Tuple[Counter, Counter]:
    """(resolved Counter[cell-key], unresolved Counter[op-name]) for one file.
    Uses a visitor (not ast.walk) so it can skip *annotation* subtrees -- otherwise
    ``List[int]`` reads as __getitem__ and ``int | None`` as __or__."""
    builtin_funcs, stdlib_funcs, modules, method_keys = idx
    alias, fromimp = {}, {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    if a.name in modules:
                        alias[a.asname] = a.name
                else:
                    # `import a.b.c` (no asname) makes a, a.b, AND a.b.c all
                    # reachable by their dotted path
                    parts = a.name.split(".")
                    for i in range(1, len(parts) + 1):
                        prefix = ".".join(parts[:i])
                        if prefix in modules:
                            alias[prefix] = prefix
        elif isinstance(n, ast.ImportFrom) and n.module in modules and not n.level:
            for a in n.names:
                if a.name in stdlib_funcs.get(n.module, ()):
                    fromimp[a.asname or a.name] = f"{n.module}.{a.name}"

    v = _OpVisitor(method_keys, builtin_funcs, stdlib_funcs, alias, fromimp, vt)
    v.visit(tree)
    return v.resolved, v.unresolved


class _OpVisitor(ast.NodeVisitor):
    def __init__(
        self,
        method_keys: DefaultDict[str, Set[str]],
        builtin_funcs: Set[str],
        stdlib_funcs: DefaultDict[str, Set[str]],
        alias: Dict[str, str],
        fromimp: Dict[str, str],
        vt: Dict[str, str],
    ) -> None:
        self.method_keys = method_keys
        self.builtin_funcs = builtin_funcs
        self.stdlib_funcs = stdlib_funcs
        self.alias, self.fromimp, self.vt = alias, fromimp, vt
        self.resolved, self.unresolved = Counter(), Counter()

    def _ambiguous(self, opname: str, recv_type: Optional[str]) -> None:
        if opname not in self.method_keys:
            return
        if recv_type:
            key = f"builtins.{recv_type}_{opname}_method"
            if key in self.method_keys[opname]:
                self.resolved[key] += 1
                return
        self.unresolved[opname] += 1

    # --- skip annotation subtrees (type expressions aren't runtime ops) ---
    def visit_AnnAssign(self, n: ast.AnnAssign) -> None:
        if n.value:
            self.visit(n.value)
        self.visit(n.target)

    def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
        for d in (
            n.args.defaults + [k for k in n.args.kw_defaults if k] + n.decorator_list
        ):
            self.visit(d)
        for s in n.body:
            self.visit(s)

    visit_AsyncFunctionDef = visit_FunctionDef

    # --- ops ---
    def visit_Call(self, n: ast.Call) -> None:
        f = n.func
        if isinstance(f, ast.Name):
            if f.id in self.fromimp:
                self.resolved[self.fromimp[f.id]] += 1
            elif f.id in self.builtin_funcs:
                self.resolved[f"builtins.{f.id}"] += 1
        elif isinstance(f, ast.Attribute):
            dotted = _dotted(f.value)
            mod = self.alias.get(dotted) if dotted else None
            if mod and f.attr in self.stdlib_funcs.get(mod, ()):
                self.resolved[f"{mod}.{f.attr}"] += 1
            elif f.attr in self.method_keys:
                self._ambiguous(f.attr, _expr_type(f.value, self.vt))
        self.generic_visit(n)

    def visit_BinOp(self, n: ast.BinOp) -> None:
        if type(n.op) in _BIN:
            self._ambiguous(
                _BIN[type(n.op)],
                _expr_type(n.left, self.vt) or _expr_type(n.right, self.vt),
            )
        self.generic_visit(n)

    def visit_AugAssign(self, n: ast.AugAssign) -> None:
        if type(n.op) in _BIN:
            self._ambiguous(
                _BIN[type(n.op)],
                _expr_type(n.target, self.vt) or _expr_type(n.value, self.vt),
            )
        self.generic_visit(n)

    def visit_UnaryOp(self, n: ast.UnaryOp) -> None:
        if type(n.op) in _UNARY and not isinstance(
            n.operand, ast.Constant
        ):  # not a -1 literal
            self._ambiguous(_UNARY[type(n.op)], _expr_type(n.operand, self.vt))
        self.generic_visit(n)

    def visit_Compare(self, n: ast.Compare) -> None:
        lt = _expr_type(n.left, self.vt)
        for op, comp in zip(n.ops, n.comparators):
            if type(op) in _CMP:
                rt = (
                    _expr_type(comp, self.vt)
                    if type(op) in (ast.In, ast.NotIn)
                    else (lt or _expr_type(comp, self.vt))
                )
                self._ambiguous(_CMP[type(op)], rt)
        self.generic_visit(n)

    def visit_Subscript(self, n: ast.Subscript) -> None:
        # skip typing generics (List[int], Optional[X], MyClass[T]) -- value is a
        # bare Capitalized name or dotted typing construct, not a runtime container
        val = n.value
        skip = isinstance(val, ast.Name) and val.id[:1].isupper()
        if not skip:
            opname = {
                "Load": "__getitem__",
                "Store": "__setitem__",
                "Del": "__delitem__",
            }.get(type(n.ctx).__name__, "__getitem__")
            self._ambiguous(opname, _expr_type(val, self.vt))
        self.generic_visit(n)


def _dotted(node: ast.AST) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _packages(corpus_dirs: Sequence[str]) -> Iterator[Tuple[str, List[Path]]]:
    """Each top-level entry of a corpus dir = one package: a dir (all .py under it)
    or a lone .py module."""
    for d in corpus_dirs:
        d = Path(d)
        for entry in sorted(d.iterdir()):
            if entry.name.startswith((".",)) or entry.name.endswith(
                (".dist-info", ".pth", ".txt")
            ):
                continue
            if entry.is_dir():
                pys = list(entry.rglob("*.py"))
                if pys:
                    yield entry.name, pys
            elif entry.suffix == ".py":
                yield entry.stem, [entry]


def mine(
    corpus_dirs: Sequence[str], idx: _Index
) -> Tuple[Dict[str, Dict[str, float]], int]:
    _bf, _sf, _mod, method_keys = idx
    # pass 1: per-package resolved/unresolved counts; accumulate resolved evidence
    per_pkg, resolved_total = [], defaultdict(float)
    for _name, files in _packages(corpus_dirs):
        rc, uc = Counter(), Counter()
        for fp in files:
            try:  # parse + visit guarded together: deep expr chains can RecursionError
                tree = ast.parse(fp.read_text(encoding="utf-8", errors="ignore"))
                r, u = file_tokens(tree, idx, _var_types(tree))
            except (Exception, RecursionError):
                continue
            rc.update(r)
            uc.update(u)
        if rc or uc:
            per_pkg.append((rc, uc))
            for k, c in rc.items():
                resolved_total[k] += c

    # type distribution per ambiguous op-name, from resolved evidence (else even)
    dist = {}
    for opname, keys in method_keys.items():
        ev = {k: resolved_total.get(k, 0.0) for k in keys}
        s = sum(ev.values())
        dist[opname] = (
            {k: ev[k] / s for k in keys}
            if s > 0
            else {k: 1.0 / len(keys) for k in keys}
        )

    # pass 2: aggregate, apportioning the unresolved remainder by dist
    sites, packages = defaultdict(float), defaultdict(float)
    for rc, uc in per_pkg:
        psite, ppres = defaultdict(float), defaultdict(float)
        for k, c in rc.items():
            psite[k] += c
            ppres[k] = 1.0
        for opname, c in uc.items():
            for k, sh in dist[opname].items():
                psite[k] += c * sh
                ppres[k] = min(1.0, ppres[k] + sh)
        for k, v in psite.items():
            sites[k] += v
        for k, v in ppres.items():
            packages[k] += v
    out = {
        k: {"sites": round(sites[k], 3), "packages": round(packages[k], 3)}
        for k in set(sites) | set(packages)
    }
    return out, len(per_pkg)


def main() -> None:
    sys.setrecursionlimit(
        16000
    )  # deep BinOp/expr chains in generated code (per-file caught anyway)
    ap = argparse.ArgumentParser(
        description="Mine a static usage prior for the support corpus."
    )
    ap.add_argument(
        "--measured",
        required=True,
        help="comma-separated measured JSON (defines cell keys)",
    )
    ap.add_argument(
        "--corpus", required=True, nargs="+", help="dir(s) of packages to scan"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    measured = {}
    for p in args.measured.split(","):
        measured.update(json.loads(Path(p.strip()).read_text()))
    idx = build_index(measured)
    usage, npkg = mine(args.corpus, idx)
    Path(args.out).write_text(json.dumps(usage, indent=2, sort_keys=True))
    covered = sum(1 for v in usage.values() if v["packages"])
    print(
        f"scanned {npkg} packages; {covered}/{len(measured)} cells have usage; wrote {args.out}"
    )
    top = sorted(usage.items(), key=lambda kv: kv[1]["packages"], reverse=True)[:25]
    print("\ntop ops by # packages:")
    for k, v in top:
        print(f"  {v['packages']:6.1f} pkgs  {int(v['sites']):7d} sites   {k}")


if __name__ == "__main__":
    main()
