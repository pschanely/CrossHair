import ast
import bz2
import collections
import math
import random

import asthelpers
import crosshair

from astunparse import unparse
import tensorflow as tf

class ProofLog(collections.namedtuple('ProofLog',['support','conclusion','proven','ms','opts'])):
    pass

s=str(ProofLog(['sos','sos'],'goal',True,1200,{}))
print(eval(s))

def load_log(log):
    with bz2.open('proof.log.bz2', 'r') as fh:
        entry = eval(fh.readline())
        entry.support = map(intern, entry.support) # support clauses are reused often!
        yield entry

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
    int: lambda br: br.choice((0,0,1,1,-1)) if br.unitrand() < 0.5 else 2**65 * br.unitrand(),
    bool: lambda br: br.choice((True, False)),
    type(None): lambda br: None,
    tuple: lambda br: tuple(gen_data(br) for _ in
        range(br.choice((0,1,2,2)) if br.unitrand()<0.5 else int(br.unitrand()*30))),
}
def gen_data(br):
    typ = br.choice(_TYPES)
    return _TYPE_GENERATORS[typ](br)

def data_to_ast(data):
    datatype = type(data)
    if datatype is tuple:
        return ast.Tuple(elts=tuple(map(data_to_ast, data)))
    elif datatype is int:
        return ast.Num(n=data)
    elif datatype is bool or data is None:
        return ast.NameConstant(value=data)
    else:
        raise Exception('unable to turn data into ast : '+repr(data))

def gen_predicate(br):
    data = data_to_ast(gen_data(br))
    print('data',data)
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

def input_fn(prover):
    seed, simplicity_bias = 1, 50
    br = BiasedRandom(seed, simplicity_bias)
    pred_ast = gen_predicate(br)
    result = crosshair_ast_eval(pred_ast)
    fdef = ast.FunctionDef(name='t', body=[ast.Return(pred_ast)])
    ret, core, extraneous = prover(fdef)
    if ret is True:
        goal_hash = semantic_hash(pred_ast)
        inputs = []
        labels = []
        for assertion, is_used in [(c, 1.0) for c in core] + [(c, 0.0) for c in extraneous]:
            inputs.append(semantic_hash(assertion) + goal_hash)
            labels.append(is_used)
        features = tf.constant(inputs).transpose()
        return {'hb{}'.format(i) : f for i,f in enumerate(features)}, labels
        # dataset = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
        # session = tf.Session()
        # session.run(batched_dataset.make_one_shot_iterator().get_next())
        # feature_dict = tf.constant()
        # return feature_dict, labels

def train(prover):
    estimator = tf.estimator.Estimator.LinearClassifier(
        feature_columns=[population, crime_rate, median_education],
    )
    # estimator = DNNClassifier(
    #     feature_columns=[sparse_feature_a_emb, sparse_feature_b_emb],
    #     hidden_units=[1024, 512, 256],
    #     optimizer=tf.train.ProximalAdagradOptimizer(
    #       learning_rate=0.1,
    #       l1_regularization_strength=0.001
    #     ))
    estimator.train(input_fn=lambda :input_fn(prover), steps=2000)
