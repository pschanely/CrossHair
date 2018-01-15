import ast
import bz2
import collections
import hashlib
import inspect
import math
import os.path
import random
import sys
import time

import asthelpers
import crosshair

from astunparse import unparse
import numpy
import tensorflow as tf

class ProofLog(collections.namedtuple('ProofLog',['support','conclusion','proven','tm','opts','kind'])):
    pass

_INTERNED = {}
def intern_string(s):
    ret = _INTERNED.get(s)
    if ret is not None:
        return ret
    _INTERNED[s] = s
    return s
def load_log(logdir):
    filename = os.path.join(logdir, 'proof.log.bz2')
    if not os.path.exists(filename):
        return
    with bz2.open(filename, 'r') as fh:
        for line in fh.readlines():
            entry = eval(line)
            yield entry._replace(support = list(map(intern_string, entry.support)))
def write_log(logdir, entries):
    with bz2.open(os.path.join(logdir, 'proof.log.bz2'), 'w') as fh:
        for entry in entries:
            fh.write(repr(entry).encode())
            fh.write('\n'.encode())

class BiasedRandom(random.Random):
    def __new__(cls, *args, **kwargs):
        return super (BiasedRandom, cls).__new__ (cls)
    def __init__(self, seed, bias_towards_zero):
        super().__init__(seed)
        self.bias = bias_towards_zero
    #
    # def __new__(cls, seed, bias_towards_zero):
    #     instance = super(BiasedRandom, cls).__new__(cls, seed)
    #     instance.bias = bias_towards_zero
    #     return instance
    def unitrand(self):
        v = super().uniform(0.0, 1.0)
        return math.pow(v, self.bias)

_TYPES = (int, tuple, bool, type(None))
_TYPE_GENERATORS = {
    int: lambda br: br.choice((0,0,1,1,-1)) if br.unitrand() < 0.5 else int(2**17 * br.unitrand()),
    bool: lambda br: br.choice((True, False)),
    type(None): lambda br: None,
    tuple: lambda br: tuple(gen_data(br) for _ in
        range(br.choice((0,1,2,2)) if br.unitrand()<0.75 else int(br.unitrand()*20))),
}

class UnableToGenerateException(Exception):
    pass

def gen_data(br, suchthat=inspect.Signature.empty):
    if suchthat is inspect.Signature.empty:
        return _TYPE_GENERATORS[br.choice(_TYPES)](br)
    else:
        for _ in range(100):
            ret = _TYPE_GENERATORS[br.choice(_TYPES)](br)
            try:
                if suchthat(ret):
                    return ret
            except:
                print(sys.exc_info())
                pass
        raise UnableToGenerateException()

def returns_true(fn, bargs):
    # print('Testing args ', bargs.arguments)
    try:
        return fn(*bargs.args, **bargs.kwargs)
    except:
        return False

def find_counterexample(assertion, min_simplicity_bias=0.75, seed=424242, timeout=0.05):
    simplicity_bias = 5
    timelimit = time.time() + timeout
    sig = inspect.signature(assertion)
    argnames = sig.parameters.keys()
    argpredicates = sig.parameters.items()
    bargs = sig.bind_partial()
    if len(argnames) == 0:
        return None if returns_true(assertion, bargs) else {}
    while True:
        #print('simplicity ', simplicity_bias)
        br = BiasedRandom(seed + int(simplicity_bias * 100000), simplicity_bias)
        try:
            for _ in range(20):
                bargs.arguments = collections.OrderedDict((n, gen_data(br, suchthat=p.annotation)) for n,p in sig.parameters.items())
                if not returns_true(assertion, bargs):
                    return bargs.arguments
        except UnableToGenerateException:
            print('unable to generate at this simplicity level')
            pass
        simplicity_bias = simplicity_bias * 0.9
        #print(simplicity_bias, min_simplicity_bias, time.time(), timelimit)
        if simplicity_bias < min_simplicity_bias or time.time() > timelimit:
            break
    return None

def data_to_ast(data):
    datatype = type(data)
    if datatype is tuple:
        return ast.Tuple(elts=tuple(map(data_to_ast, data)),ctx=ast.Load())
    elif datatype is int:
        return ast.Num(n=data)
    elif datatype is bool or data is None:
        return ast.NameConstant(value=data)
    else:
        raise Exception('unable to turn data into ast : '+repr(data))

def gen_predicate(br):
    data = data_to_ast(gen_data(br))
    # print('data',data)
    names = ('isint', 'istuple', 'isdefined', 'isbool', 'isnone')
    return ast.Call(func=ast.Name(id=br.choice(names)), args=[data], keywords=())

def crosshair_ast_eval(node):
    return eval(unparse(node), crosshair.__dict__)
    # ast.Module(
    #     statements=[
    #         ast.ImportFrom('crosshair', '*', 0),
    #         ast.Expr
    #     ]
    # )

def _hash(s):
    h = hashlib.sha256()
    h.update(s.encode())
    return h

def _deterministic_hash(stringdatums):
    s = ''.join(stringdatums)
    return int.from_bytes(_hash(s).digest(), byteorder='big')

def _fn_object_name(fn):
    fntype = type(fn)
    if fntype == ast.Name:
        return fn.id
    elif fntype == ast.FunctionDef:
        return fn.name
    else:
        return type(fn).__name__

_bin_ops = [
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult
]
NUM_SEMANTIC_HASH_BUCKETS = 211
class SemanticAstHasher(asthelpers.ScopeTracker):
    def __init__(self):
        super().__init__()
        self.buckets = numpy.zeros((NUM_SEMANTIC_HASH_BUCKETS,))
        self._stoptypes = set([ast.Num, ast.Name, ast.NameConstant])
        self._skiptypes = set([ast.Load] + _bin_ops)
    def root_label(self, node):
        nodetype = type(node)
        if nodetype == ast.BinOp:
            return [node.op.__class__.__name__]
        elif nodetype == ast.Call:
            return self.root_label(node.func)
            target = self.resolve(node.func)
            # print('call target', unparse(target))
            return [_fn_object_name(target)]
        elif nodetype == ast.Name:
            target = self.resolve(node)
            # print(node.id,' resolve to ',type(target), target)
            if target is not node:
                return ['uuu']
            return [node.id]
        else:
            return [node.__class__.__name__]
    # def visit_BinOp(self, node):
    #     return self.generic_visit(node, [node.op.__class__.__name__])
    # def visit_Call(self, node):
    #     target = self.resolve(node.func)
    #     name = target.name if type(target) == ast.FunctionDef else target.__class__.__name__
    #     return self.generic_visit(node, [name])
    # def visit_Lambda(self, node):
    #     return super().generic_visit(node)
    def generic_visit(self, node):
        if type(node) in self._skiptypes: return node
        hashsrc = self.root_label(node)
        self.add(hashsrc) # for the unary registration
        if type(node) in self._stoptypes:
            return node
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                value = (value,)
            for (idx, item) in enumerate(value):
                if isinstance(item, ast.AST):
                    if type(item) in self._skiptypes: continue
                    self.visit(item)
                    self.add(hashsrc + [field, str(idx)] + self.root_label(item))
        return node
    def add(self, hashsrc):
        val = _deterministic_hash(hashsrc)
        # print('  ', val % len(self.buckets), ' <= ', hashsrc)
        self.buckets[val % len(self.buckets)] = 1.0

def semantic_hash(astobject):
    # print('semantic_hash', unparse(astobject))
    h = SemanticAstHasher()
    h.add(['root'] + h.root_label(astobject))
    h.visit(astobject)
    return h.buckets

def utility_score(axiom, proof):
    # lower scores are better
    unrelated_axiom_bias = numpy.maximum(axiom - proof, 0.0)
    matching_bias = -(axiom * proof)
    # print('utility ', sum(unrelated_axiom_bias), sum(matching_bias))
    return sum(unrelated_axiom_bias) + sum(matching_bias) * 3

def order_axioms(axioms, proof_obligation):
    proof_semantic = semantic_hash(proof_obligation)
    scores_and_axioms = sorted(
        ((utility_score(semantic_hash(axiom), proof_semantic), axiom)
        for axiom in axioms),
        key= lambda p:p[0]
    )
    return [a for (s, a) in scores_and_axioms]


def find_constant_predicate_examples(prover, oracle, seed, simplicity_bias=50, iterations=10):
    print('LEN', len(oracle.prooflog))
    print(oracle.prooflog.items())
    br = BiasedRandom(seed, simplicity_bias)
    for _ in range(iterations):
        pred_ast = gen_predicate(br)
        result = crosshair_ast_eval(pred_ast)
        if result is not True: continue
        fdef = ast.FunctionDef(name='t', body=[ast.Return(pred_ast)], decorator_list=[], args=ast.arguments(args=[], defaults=[], vararg=None, kwarg=[], kwonlyargs=[]))
        if oracle.log_contains_proof(fdef): continue
        t0 = time.time()
        ret, support = prover(fdef)
        if ret is not True:
            raise Exception('unable to prove true predicate over constant')
        print('added',unparse(fdef))
        oracle.add_to_log(ProofLog(conclusion=unparse(fdef), support=support, proven=ret, tm=time.time()-t0, opts={}, kind='constant_predicate'))
    print('LEN', len(oracle.prooflog))

class TrivialProofOracle:
    def score_axioms(self, axioms, goal):
        for axiom in axioms:
            axiom.set_score(100)

class ProofOracle:
    def __init__(self, prooflogdir):
        self.prooflogdir = prooflogdir
        self.prooflog = {}
        for entry in load_log(prooflogdir):
            self.prooflog[entry.conclusion] = entry
        self.model = None
    def add_to_log(self, entry):
        # print(entry)
        self.prooflog[entry.conclusion] = entry
    def log_contains_proof(self, goal):
        return unparse(goal) in self.prooflog
    def save_log(self):
        write_log(self.prooflogdir, self.prooflog.values())
    def train(self, prover):
        keys = [tf.feature_column.numeric_column('hb{}'.format(i))
            for i in range(asthelpers.NUM_SEMANTIC_HASH_BUCKETS)]
        # estimator = tf.estimator.Estimator.LinearClassifier(
        #     feature_columns=keys,
        # )
        estimator = tf.estimator.DNNClassifier(
            feature_columns=keys,
            hidden_units=[32, 16],
            n_classes=2,
            optimizer=tf.train.ProximalAdagradOptimizer(
              learning_rate=0.1,
              l1_regularization_strength=0.001
            ))

        estimator.train(input_fn=input_fn(prover, 42, num_epochs=None, shuffle=True), steps=200)
        print('trained?')

        scores = estimator.evaluate(input_fn=input_fn(prover, 43, num_epochs=1, shuffle=False))

        print("\nTest Scores:", scores)

    def score_axioms(self, axioms, goal):
        goal_hash = semantic_hash(goal)
        for axiom in axioms:
            score = 100
            if axiom.src_ast:
                utility_score(semantic_hash(axiom.src_ast), goal_hash)
            axiom.set_score(score)

def input_fn(prover, seed, **kw):
    simplicity_bias = 50
    br = BiasedRandom(seed, simplicity_bias)
    inputs = []
    labels = []
    while True:
        pred_ast = gen_predicate(br)
        result = crosshair_ast_eval(pred_ast)
        fdef = ast.FunctionDef(name='t', body=[ast.Return(pred_ast)], decorator_list=[], args=ast.arguments(args=[], defaults=[], vararg=None, kwarg=[], kwonlyargs=[]))
        ret, core, extraneous = prover(fdef)
        if ret is True:
            print('yielding {} positive and {} negative examples'.format(len(core), len(extraneous)))
            goal_hash = asthelpers.semantic_hash(fdef)
            for assertion, is_used in [(c, 1.0) for c in core] + [(c, 0.0) for c in extraneous]:
                if assertion.src_ast is None: continue
                inputs.append(asthelpers.semantic_hash(assertion.src_ast) + goal_hash)
                labels.append(is_used)
            if len(inputs) > 25:
                break
    features = numpy.asarray(inputs).transpose()
    return tf.estimator.inputs.numpy_input_fn(
        x={'hb{}'.format(i) : f for i,f in enumerate(features)},
        y=numpy.array(labels), **kw)

    # dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
    # session = tf.Session()
    # session.run(batched_dataset.make_one_shot_iterator().get_next())
    # feature_dict = tf.constant()
    # return feature_dict, labels
