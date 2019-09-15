import collections
import copy
import math
import unittest

from crosshair.core import *
import crosshair.examples.arith
import crosshair.examples.tic_tac_toe
from crosshair import contracted_builtins
from crosshair.util import set_debug




#
# Begin fixed line number area.
# Tests depend on the line number of the following section.
#

class Pokeable:
    '''
    inv: self.x >= 0
    '''
    x :int = 1
    def poke(self) -> None:
        '''
        post[self]: True
        '''
        self.x += 1
    def wild_pokeby(self, amount:int) -> None:
        '''
        post[self]: True
        '''
        self.x += amount
    def safe_pokeby(self, amount:int) -> None:
        '''
        pre: amount >= 0
        post[self]: True
        '''
        self.x += amount
    def __repr__(self) -> str:
        return 'Pokeable(' + repr(self.x) + ')'
    def __init__(self, x:int) -> None:
        '''
        pre: x >= 0
        '''
        self.x = x


#
# End fixed line number area.
#

class PersonTuple(NamedTuple):
    name: str
    age: int


NOW = 1000
@dataclass
class Person:
    name: str
    birth: int
    def __init__(self):
        raise Exception('I cannot be created!')
    
    def _getage(self):
        return NOW - self.birth
    def _setage(self, newage):
        self.birth = NOW - newage
    def _delage(self):
        del self.birth
    age = property(_getage, _setage, _delage, 'Age of person')

class Color(enum.Enum):
    RED = 0
    BLUE = 1
    GREEN = 2

class SmokeDetector:
    ''' inv: not (self._is_plugged_in and self._in_original_packaging) '''
    _in_original_packaging: bool
    _is_plugged_in: bool
    def signaling_alarm(self, air_samples: List[str]) -> bool:
        '''
        pre: self._is_plugged_in
        post: implies('smoke' in air_samples, _ == True)
        '''
        return 'smoke' in air_samples

class Measurer:
    def measure(self, x: int) -> str:
        '''
        post: _ == self.measure(-x)
        '''
        return 'small' if x <= 10 else 'large'

def fibb(x:int) -> int:
    '''
    pre: x>=0
    post: _ < 10
    '''
    if x <= 2:
        return 1
    r1,r2 = fibb(x-1), fibb(x-2)
    ret = r1 + r2
    return ret

def recursive_example(x:int) -> bool:
    '''
    pre: x >= 0
    post:
        __old__.x >= 0  # just to confirm __old__ works in recursive cases
        _ == True
    '''
    if x == 0:
        return True
    else:
        return recursive_example(x - 1)


def check_fail(fn):
    return ([m.state for m in analyze_function(fn)], [MessageType.POST_FAIL])

def check_exec_err(fn):
    return ([m.state for m in analyze_function(fn)], [MessageType.EXEC_ERR])

def check_post_err(fn):
    return ([m.state for m in analyze_function(fn)], [MessageType.POST_ERR])

def check_unknown(fn):
    return ([(m.state, m.message, m.traceback) for m in analyze_function(fn)],
            [(MessageType.CANNOT_CONFIRM, 'I cannot confirm this', '')])

def check_ok(fn):
    return (analyze_function(fn), [])

def check_messages(msgs, **kw):
    default_msg = AnalysisMessage(MessageType.CANNOT_CONFIRM, '', '', 0, 0, '')
    msg = msgs[0] if msgs else replace(default_msg)
    fields = ('state','message','filename','line','column','traceback', 'execution_log')
    for k in fields:
        if k not in kw:
            default_val = getattr(default_msg, k)
            msg = replace(msg, **{k: default_val})
            kw[k] = default_val
    if msgs:
        msgs[0] = msg
    return (msgs, [AnalysisMessage(**kw)])


class ProxiedObjectTest(unittest.TestCase):
    def test_copy(self) -> None:
        poke1 = ProxiedObject(StateSpace(), Pokeable, 'ppoke')
        poke1.poke()
        poke2 = copy.copy(poke1)
        self.assertIsNot(poke1, poke2)
        self.assertEqual(type(poke1), type(poke2))
        self.assertIs(poke1.x, poke2.x)
        poke1.poke()
        self.assertIsNot(poke1.x, poke2.x)
        self.assertNotEqual(str(poke1.x.var), str(poke2.x.var))

    def test_proxy_alone(self) -> None:
        def f(pokeable :Pokeable) -> None:
            '''
            post[pokeable]: pokeable.x > 0
            '''
            pokeable.poke()
        self.assertEqual(*check_ok(f))

    def test_proxy_in_list(self) -> None:
        def f(pokeables :List[Pokeable]) -> None:
            '''
            pre: len(pokeables) == 1
            post: all(p.x > 0 for p in pokeables)
            '''
            for pokeable in pokeables:
                pokeable.poke()
        self.assertEqual(*check_ok(f))


class BooleanTest(unittest.TestCase):

    def test_simple_bool_with_fail(self) -> None:
        def f(a:bool, b:bool) -> bool:
            ''' post: _ == a '''
            return True if a else b
        self.assertEqual(*check_fail(f))

    def test_simple_bool_ok(self) -> None:
        def f(a:bool, b:bool) -> bool:
            ''' post: _ == a or b '''
            return True if a else b
        self.assertEqual(*check_ok(f))
        
    def test_bool_ors_fail(self) -> None:
        def f(a:bool, b:bool, c:bool, d:bool) -> bool:
            ''' post: _ == (a ^ b) or (c ^ d) '''
            return a or b or c or d
        self.assertEqual(*check_fail(f))
        
    def test_bool_ors(self) -> None:
        def f(a:bool, b:bool, c:bool, d:bool) -> bool:
            '''
            pre: (not a) and (not d)
            post: _ == (a ^ b) or (c ^ d)
            '''
            return a or b or c or d
        self.assertEqual(*check_ok(f))
        
class NumbersTest(unittest.TestCase):
    
    def test_numeric_promotions(self) -> None:
        def f(b:bool, i:int) -> Tuple[int, float, float]:
            '''
            #post: 100 <= _[0] <= 101
            #post: 3.14 <= _[1] <= 4.14
            post: isinstance(_[2], float)
            '''
            return ((b + 100), (b + 3.14), (i + 3.14))
        self.assertEqual(*check_ok(f))

    def test_int_reverse_operators(self) -> None:
        def f(i:int) -> float:
            '''
            pre: i != 0
            post: _ > 0
            '''
            return (1 + i) + (1 - i) + (1 / i)
        self.assertEqual(*check_ok(f))

    def test_int_div_fail(self) -> None:
        def f(a:int, b:int) -> int:
            ''' post: a <= _ <= b '''
            return (a + b) // 2
        self.assertEqual(*check_fail(f))

    def test_int_div_ok(self) -> None:
        def f(a:int, b:int) -> int:
            '''
            pre: a < b
            post: a <= _ <= b
            '''
            return (a + b) // 2
        self.assertEqual(*check_ok(f))

    def test_int_bitwise_fail(self) -> None:
        def f(a:int, b:int) -> int:
            '''
            pre: 0 <= a <= 3
            pre: 0 <= b <= 3
            post: _ < 7
            '''
            return (a << 1) ^ b
        self.assertEqual(*check_fail(f))
        
    def test_int_bitwise_ok(self) -> None:
        def f(a:int, b:int) -> int:
            '''
            pre: 0 <= a <= 3
            pre: 0 <= b <= 3
            post: _ <= 7
            '''
            return (a << 1) ^ b
        self.assertEqual(*check_ok(f))
        
    def test_true_div_fail(self) -> None:
        def f(a:int, b:int) -> float:
            '''
            pre: a != 0 and b != 0
            post: _ >= 1.0
            '''
            return (a + b) / b
        self.assertEqual(*check_fail(f))
    
    def test_true_div_ok(self) -> None:
        def f(a:int, b:int) -> float:
            '''
            pre: a >= 0 and b > 0
            post: _ >= 1.0
            '''
            return (a + b) / b
        self.assertEqual(*check_ok(f))

    def test_trunc_fail(self) -> None:
        def f(n:float) -> int:
            '''
            pre: n > 100
            post: _ < n
            '''
            return math.trunc(n)
        self.assertEqual(*check_fail(f))
        
    def test_trunc_ok(self) -> None:
        def f(n:float) -> int:
            ''' post: abs(_) <= abs(n) '''
            return math.trunc(n)
        self.assertEqual(*check_ok(f))
        
    def test_round_fail(self) -> None:
        def f(n1:int, n2:int) -> Tuple[int,int]:
            '''
            pre: n1 < n2
            post: _[0] < _[1] # because we round towards even
            '''
            return (round(n1+0.5), round(n2+0.5))
        self.assertEqual(*check_fail(f))
        
    def test_round_unknown(self) -> None:
        def f(num:float, ndigits:Optional[int]) -> float:
            '''
            post: isinstance(_, int) == (ndigits is None)
            '''
            return round(num, ndigits)
        self.assertEqual(*check_unknown(f))  # TODO: this is unknown (z3 can't solve 10**x != 0 right now)

    def test_number_isinstance(self) -> None:
        def f(x:float) -> float:
            ''' post: isinstance(_, float) '''
            return x
        self.assertEqual(*check_ok(f))
        
class StringsTest(unittest.TestCase):

    def test_cast_to_bool_fail(self) -> None:
        def f(a:str) -> str:
            ''' post: a '''
            return a
        self.assertEqual(*check_fail(f))
    
    def test_multiply_fail(self) -> None:
        def f(a:str) -> str:
            ''' post: len(_) == len(a) * 3 '''
            return 3 * a
        self.assertEqual(*check_ok(f))

    def TODO_supported_in_head_test_compare_ok(self) -> None:
        def f(a:str, b:str) -> bool:
            ''' post: True '''
            return a < b
        self.assertEqual(*check_ok(f))

    def test_multiply_ok(self) -> None:
        def f(a:str) -> str:
            ''' post: len(_) == len(a) * 5 '''
            return a * 3 + 2 * a
        self.assertEqual(*check_ok(f))
    
    def test_prefixing_fail(self) -> None:
        def f(a:str, indent:bool) -> str:
            ''' post: len(_) == len(a) + indent '''
            return ('  ' if indent else '') + a
        self.assertEqual(*check_fail(f))
    
    def test_prefixing_ok(self) -> None:
        def f(a:str, indent:bool) -> str:
            ''' post: len(_) == len(a) + (2 if indent else 0) '''
            return ('  ' if indent else '') + a
        self.assertEqual(*check_ok(f))

    def test_negative_index_slicing(self) -> None:
        def f(s:str) -> Tuple[str, str]:
            ''' post: sum(map(len, _)) == len(s) - 1 '''
            idx = s.find(':')
            return (s[:idx], s[idx+1:])
        self.assertEqual(*check_fail(f))  # (fails when idx == -1)

    def test_int_str_comparison_fail(self) -> None:
        def f(a:int, b:str) -> Tuple[bool, bool]:
            ''' post: (not _[0]) or (not _[1]) '''
            return (a != b, b != a)
        self.assertEqual(*check_fail(f))

    def test_int_str_comparison_ok(self) -> None:
        def f(a:int, b:str) -> bool:
            ''' post: _ == False '''
            return a == b or b == a
        self.assertEqual(*check_ok(f))

    def test_string_formatting_literal(self) -> None:
        def f(o:object) -> str:
            ''' post: True '''
            return 'object of type {typ} with repr {zzzzz}'.format(typ=type(o), rep=repr(o))
        self.assertEqual(*check_exec_err(f))
    
    def test_string_formatting_varfmt(self) -> None:
        def f(fmt:str) -> str:
            '''
            # NOTE: with a iteration-base, pure python implementation of format, we wouldn't need this precondition:
            pre: '{}' in fmt
            post: True
            '''
            return fmt.format(ver=sys.version, platform=sys.platform)
        self.assertEqual(*check_exec_err(f))
    
class TuplesTest(unittest.TestCase):
        
    def test_tuple_range_intersection_fail(self) -> None:
        def f(a:Tuple[int, int], b:Tuple[int, int]) -> Optional[Tuple[int, int]]:
            '''
            pre: a[0] < a[1] and b[0] < b[1]
            post: _[0] <= _[1]
            '''
            return (max(a[0], b[0]), min(a[1], b[1]))
        self.assertEqual(*check_fail(f))
    
    def test_tuple_range_intersection_ok(self) -> None:
        def f(a:Tuple[int, int], b:Tuple[int, int]) -> Optional[Tuple[int, int]]:
            '''
            pre: a[0] < a[1] and b[0] < b[1]
            post: _ is None or _[0] <= _[1]
            '''
            if a[1] > b[0] and a[0] < b[1]: # (if the ranges overlap)
                return (max(a[0], b[0]), min(a[1], b[1]))
            else:
                return None
        self.assertEqual(*check_ok(f))

    def test_tuple_with_uniform_values_fail(self) -> None:
        def f(a:Tuple[int, ...]) -> float:
            '''
            post: True
            '''
            return sum(a) / len(a)
        self.assertEqual(*check_exec_err(f))
        
    def test_tuple_with_uniform_values_ok(self) -> None:
        def f(a:Tuple[int, ...]) -> Tuple[int, ...]:
            '''
            pre: len(a) < 4
            post: 0 not in _
            '''
            return tuple(x for x in a if x)
        self.assertEqual(*check_ok(f))
        
class ListsTest(unittest.TestCase):
    
    def test_containment_fail(self) -> None:
        def f(a:int, b:List[int]) -> bool:
            '''
            post: _ == (a in b[:5])
            '''
            return a in b
        self.assertEqual(*check_fail(f))
        
    def test_containment_ok(self) -> None:
        def f(a:int, b:List[int]) -> bool:
            '''
            pre: 1 == len(b)
            post: _ == (a == b[0])
            '''
            return a in b
        self.assertEqual(*check_ok(f))
        
    def test_doubling_fail(self) -> None:
        def f(a:List[int]) -> List[int]:
            '''
            post: len(_) > len(a)
            '''
            return a + a
        self.assertEqual(*check_fail(f))

    def test_doubling_ok(self) -> None:
        def f(a:List[int]) -> List[int]:
            '''
            post: len(_) > len(a) or not a
            '''
            return a + a
        self.assertEqual(*check_ok(f))

    def test_mixed_symbolic_and_literal_concat_ok(self) -> None:
        def f(l:List[int], i:int) -> List[int]:
            '''
            pre: i >= 0
            post: len(_) == len(l) + 1
            '''
            return l[:i] + [42,] + l[i:]
        self.assertEqual(*check_ok(f))

    def test_range_fail(self) -> None:
        def f(l:List[int]) -> List[int]:
            '''
            pre: len(l) == 3
            post: len(_) > len(l)
            '''
            n:List[int] = [] 
            for i in range(len(l)):
                n.append(l[i] + 1)
            return n
        self.assertEqual(*check_fail(f))
    
    def test_range_ok(self) -> None:
        def f(l:List[int]) -> List[int]:
            '''
            pre: l and len(l) < 10  # (max is to cap runtime)
            post: _[0] == l[0] + 1
            '''
            n:List[int] = [] 
            for i in range(len(l)):
                n.append(l[i] + 1)
            return n
        self.assertEqual(*check_ok(f))

    def test_extend_literal_unknown(self) -> None:
        def f(l:List[int]) -> List[int]:
            '''
            post: _[:2] == [1, 2]
            '''
            r = [1,2,3]
            r.extend(l)
            return r
        self.assertEqual(*check_unknown(f))
            
    def test_index_error(self) -> None:
        def f(l:List[int], idx:int) -> int:
            '''
            pre: idx >= 0 and len(l) > 2
            post: True
            '''
            return l[idx]
        self.assertEqual(*check_exec_err(f))
            
    def test_nested_lists_fail(self) -> None:
        def f(l:List[List[int]]) -> int:
            '''
            post: _ > 0
            '''
            total = 0
            for i in l:
                total += len(i)
            return total
        self.assertEqual(*check_fail(f))
        
    def test_nested_lists_ok(self) -> None:
        def f(l:List[List[int]]) -> int:
            '''
            pre: len(l) < 4
            post: _ >= 0
            '''
            total = 0
            for i in l:
                total += len(i)
            return total
        self.assertEqual(*check_ok(f))
        
    def test_slice_outside_range_ok(self) -> None:
        def f(l:List[int], i:int)->List[int]:
            '''
            pre: i >= len(l)
            post: _ == l
            '''
            return l[:i]
        self.assertEqual(*check_ok(f))
        
    def test_slice_amount(self) -> None:
        def f(l:List[int])->List[int]:
            '''
            pre: len(l) >= 3
            post: len(_) == 1
            '''
            return l[2:3]
        self.assertEqual(*check_ok(f))
        
    def test_slice_assignment_ok(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) >= 4
            post[l]:
                l[1] == 42
                l[2] == 43
                #len(l) == 3 # TODO
            '''
            l[1:-1] = [42, 43] # TODO: when I change this, I get POST_FAIL and CANNOT_CONFIRM
        self.assertEqual(*check_ok(f))
        
    def test_insert_ok(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) == 4
            post[l]:
                len(l) == 5
                l[2] == 42
            '''
            l.insert(-2, 42)
        self.assertEqual(*check_ok(f))
        
    def test_assignment_ok(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) >= 4
            post[l]: l[3] == 42
            '''
            l[3] = 42
        self.assertEqual(*check_ok(f))
        
    def test_slice_delete_fail(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) >= 2
            post[l]: len(l) > 0
            '''
            del l[-2:]
        self.assertEqual(*check_fail(f))

    def test_item_delete_ok(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) == 5
            post[l]: len(l) == 4
            '''
            del l[2]
        self.assertEqual(*check_ok(f))
        
    def test_sort_ok(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) == 3
            post[l]: l[0] == min(l)
            '''
            l.sort()
        self.assertEqual(*check_ok(f))
        
    def test_reverse_ok(self) -> None:
        def f(l:List[int])->None:
            '''
            pre: len(l) == 3
            post[l]: l[0] == 42
            '''
            l.append(42)
            l.reverse()
        self.assertEqual(*check_ok(f))
        
class DictionariesTest(unittest.TestCase):
    
    def test_dict_basic_fail(self) -> None:
        def f(a:Dict[int, str], k:int, v:str) -> None:
            '''
            post[a]: a[k] == "beep"
            '''
            a[k] = v
        self.assertEqual(*check_fail(f))

    def test_dict_basic_ok(self) -> None:
        def f(a:Dict[int, str], k:int, v:str) -> None:
            '''
            post[a]: a[k] == v
            '''
            a[k] = v
        self.assertEqual(*check_ok(f))

    def test_dict_empty_bool(self) -> None:
        def f(a:Dict[int, str]) -> bool:
            '''
            post[a]: _ == True
            '''
            a[0] = 'zero'
            return bool(a)
        self.assertEqual(*check_ok(f))

    def test_dict_over_objects(self) -> None:
        def f(a: Dict[object, object]) -> int:
            '''
            post: _ >= 0
            '''
            return len(a)
        self.assertEqual(*check_ok(f))
        
    def test_dict_iter_fail(self) -> None:
        def f(a:Dict[int, str]) -> List[int]:
            '''
            post[a]: 5 in _
            '''
            a[10] = 'ten'
            return list(a.__iter__())
        self.assertEqual(*check_fail(f))

    def test_dict_iter_ok(self) -> None:
        def f(a:Dict[int, str]) -> List[int]:
            '''
            pre: len(a) < 4
            post[a]: 10 in _
            '''
            a[10] = 'ten'
            return list(a.__iter__())
        self.assertEqual(*check_ok(f))

    def test_dict_to_string_ok(self) -> None:
        def f(a:Dict[int, str]) -> str:
            '''
            pre: len(a) == 0
            post: _ == '{}'
            '''
            return str(a)
        self.assertEqual(*check_ok(f))
    
    def test_dict_items_ok(self) -> None:
        def f(a:Dict[int, str]) -> Iterable[Tuple[int,str]]:
            '''
            pre: len(a) < 5
            post[a]: (10,'ten') in _
            '''
            a[10] = 'ten'
            return a.items()
        self.assertEqual(*check_ok(f))

    def test_dict_del_fail(self) -> None:
        def f(a:Dict[str, int]) -> None:
            '''
            post[a]: True
            '''
            del a["42"]
        self.assertEqual(*check_exec_err(f))

    def test_dicts_complex_contents(self) -> None:
        def f(d: Dict[Tuple[int, str], Tuple[float, int]]) -> int:
            '''
            post: _ > 0
            '''
            if (42, 'fourty-two') in d:
                return d[(42, 'fourty-two')][1]
            else:
                return 42
        self.assertEqual(*check_fail(f))
    
    def test_dicts_subtype_lookup(self) -> None:
        def f(d: Dict[Tuple[int, str], int]) -> None:
            '''
            pre: not d
            post[d]: [(42, 'fourty-two')] == list(d.keys())
            '''
            d[(42, 'fourty-two')] = 1
        self.assertEqual(*check_ok(f))
    
    def test_dicts_complex_keys(self) -> None:
        # TODO: local fn here isn't callable from postcondition
        def f(dx: Dict[Tuple[int, str], int]) -> None:
            '''
            pre: not dx
            post[dx]:
                len(dx) == 1
                dx[(42, 'fourty-two')] == 1
            '''
            dx[(42, 'fourty-two')] = 1
            #dx[(40 + 2, 'fourty' + '-two')] = 2
        self.assertEqual(*check_ok(f))
    
    def test_dicts_inside_lists(self) -> None:
        def f(dicts:List[Dict[int, int]]) -> Dict[int, int]:
            '''
            pre: len(dicts) <= 1  # to narrow search space (would love to make this larger)
            post: len(_) <= len(dicts)
            '''
            ret = {}
            for d in dicts:
                ret.update(d)
            return ret
        self.assertEqual(*check_fail(f))
    
    def test_dicts_inside_lists_with_identity(self) -> None:
        # TODO: the message is confusing because it reflects the "after" state
        # and because identity is lost in the repr(). ("fails when f = [{}, {}]")
        def f(dicts:List[Dict[int, int]]):
            '''
            Removes duplicate keys.
            pre: len(dicts) == 2
            pre:  len(dicts[0]) == 1
            post: len(dicts[0]) == 1
            '''
            seen :Set[int] = set()
            for d in dicts:
                for k in d.keys():
                    if k in seen:
                        del d[k]
                    else:
                        seen.add(k)
        self.assertEqual(*check_fail(f))
    

class SetsTest(unittest.TestCase):
    
    def test_basic_fail(self) -> None:
        def f(a:Set[int], k:int) -> None:
            '''
            post[a]: k+1 in a
            '''
            a.add(k)
        self.assertEqual(*check_fail(f))

    def test_basic_ok(self) -> None:
        def f(a:Set[int], k:int) -> None:
            '''
            post[a]: k in a
            '''
            a.add(k)
        self.assertEqual(*check_ok(f))

    def test_union_fail(self) -> None:
        def f(a:Set[str], b:Set[str]) -> Set[str]:
            '''
            post: all(((i in a) and (i in b)) for i in _)
            '''
            return a | b
        self.assertEqual(*check_fail(f))

    def test_union_ok(self) -> None:
        def f(a:Set[str], b:Set[str]) -> Set[str]:
            '''
            post: all(((i in a) or (i in b)) for i in _)
            '''
            return a | b
        self.assertEqual(*check_unknown(f))

class ProtocolsTest(unittest.TestCase):
    def test_hashable_values_fail(self) -> None:
        def f(b:bool, i:int, t:Tuple[str, ...], s:FrozenSet[float]) -> int:
            ''' post: _ % 10 != 0 '''
            return hash((i, t, s))
        self.assertEqual(*check_fail(f))

    def test_hashable_values_ok(self) -> None:
        def f(a:Tuple[str, int, float, bool],
              b:Tuple[str, int, float, bool]) -> int:
            ''' post: _ or not (a == b) '''
            return hash(a) == hash(b)
        self.assertEqual(*check_unknown(f))


    def test_symbolic_hashable(self) -> None:
        def f(a:Hashable) -> int:
            ''' post: 0 <= _ <= 1 '''
            return hash(a) % 2
        self.assertEqual(*check_ok(f))

    def test_symbolic_supports(self) -> None:
        def f(a:SupportsAbs, f:SupportsFloat, i:SupportsInt, r:SupportsRound, c:SupportsComplex, b:SupportsBytes) -> float:
            ''' post: _.real <= 0 '''
            return abs(a) + float(f) + int(i) + round(r) + complex(c) + len(bytes(b))
        self.assertEqual(*check_fail(f))

    def test_iterable(self) -> None:
        T = TypeVar('T')
        def f(a:Iterable[T]) -> T:
            '''
            pre: a
            post: _ in a
            '''
            return next(iter(a))
        self.assertEqual(*check_ok(f))

    def test_bare_type(self) -> None:
        def f(a:List) -> bool:
            '''
            pre: a
            post: _
            '''
            return bool(a)
        self.assertEqual(*check_ok(f))

    # TODO: though, most bare types don't work yet (because <Any> doesn't work?)

class EnumsTest(unittest.TestCase):

    def test_enum_identity_matches_equality(self) -> None:
        def f(color1:Color, color2:Color) -> bool:
            ''' post: _ == (color1 is color2) '''
            return color1 == color2
        self.assertEqual(*check_ok(f))

    def TODO_test_enum_in_container(self) -> None:
        # TODO: unknown sat for this one currently; see
        # https://stackoverflow.com/questions/57404130/tactics-for-z3-sequence-problems
        # update: z3 at head deals with this correctly
        def f(colors :List[Color]) -> bool:
            ''' post: not _ '''
            return Color.RED in colors and Color.BLUE in colors
        self.assertEqual(*check_fail(f))

class ObjectsTest(unittest.TestCase):

    def test_obj_member_fail(self) -> None:
        def f(foo:Pokeable) -> int:
            '''
            pre: 0 <= foo.x <= 4
            post[foo]: _ < 5
            '''
            foo.poke()
            foo.poke()
            return foo.x
        self.assertEqual(*check_fail(f))

    def test_obj_member_nochange_ok(self) -> None:
        def f(foo:Pokeable) -> int:
            ''' post: _ == foo.x '''
            return foo.x
        self.assertEqual(*check_ok(f))

    def test_obj_member_change_ok(self) -> None:
        def f(foo:Pokeable) -> int:
            '''
            pre: foo.x >= 0
            post[foo]: foo.x >= 2
            '''
            foo.poke()
            foo.poke()
            return foo.x
        self.assertEqual(*check_ok(f))

    def test_obj_member_change_detect(self) -> None:
        def f(foo:Pokeable) -> int:
            '''
            pre: foo.x > 0
            post: True
            '''
            foo.poke()
            return foo.x
        self.assertEqual(*check_post_err(f))

    def test_example_second_largest(self) -> None:
        def second_largest(items: List[int]) -> int:
            '''
            pre: 2 <= len(items) <= 3  # (max is to cap runtime)
            post: _ == sorted(items)[-2]
            '''
            next_largest, largest = items[:2]
            if largest < next_largest:
                next_largest, largest = largest, next_largest
                
            for item in items[2:]:
                if item > largest:
                    largest, next_largest = (item, largest)
                elif item > next_largest:
                    next_largest = item
            return next_largest
        self.assertEqual(*check_ok(second_largest))

    def test_class(self) -> None:
        messages = analyze_class(Pokeable)
        self.assertEqual(*check_messages(messages,
                                         state=MessageType.POST_FAIL,
                                         filename='crosshair/core_test.py',
                                         line=30,
                                         column=0))

    def test_extend_namedtuple(self) -> None:
        def f(p: PersonTuple) -> PersonTuple:
            '''
            post: _.age != 222
            '''
            return PersonTuple(p.name, p.age + 1)
        self.assertEqual(*check_fail(f))

    def test_property(self) -> None:
        def f(p: Person) -> None:
            '''
            pre: 0 <= p.age < 100
            post[p]: p.birth + p.age == NOW
            '''
            assert p.age == NOW - p.birth
            oldbirth = p.birth
            p.age = p.age + 1
            assert oldbirth == p.birth + 1
        self.assertEqual(*check_ok(f))

    def test_typevar(self) -> None:
        T = TypeVar('T')
        class MaybePair(Generic[T]):
            '''
            inv: (self.left is None) == (self.right is None)
            '''
            left :Optional[T]
            right :Optional[T]
            def setpair(self, left:Optional[T], right:Optional[T]):
                '''post[self]: True'''
                if (left is None) ^ (right is None):
                    raise ValueError('one side must be given if the other is')
                self.left, self.right = left, right
        messages = analyze_class(MaybePair)
        self.assertEqual(*check_messages(messages, state=MessageType.EXEC_ERR))

    def test_bad_invariant(self):
        class Foo:
            '''
            inv: self.item == 7
            '''
            def do_a_thing(self) -> None:
                pass
        self.assertEqual(*check_messages(analyze_class(Foo), state=MessageType.PRE_UNSAT))

    def test_inheritance_base_class_ok(self):
        self.assertEqual(analyze_class(SmokeDetector), [])

    def test_use_inherited_postconditions(self):
        class CarbonMonoxideDetector(SmokeDetector):
            def signaling_alarm(self, air_samples: List[str]) -> bool:
                '''
                post: implies('carbon_monoxide' in air_samples, _ == True)
                '''
                return 'carbon_monoxide' in air_samples  # fails: does not detect smoke
        self.assertEqual(*check_messages(analyze_class(CarbonMonoxideDetector),
                                         state=MessageType.POST_FAIL))

    def test_inherited_preconditions_overridable(self):
        class SmokeDetectorWithBattery(SmokeDetector):
            _battery_power: int
            def signaling_alarm(self, air_samples: List[str]) -> bool:
                '''
                pre: self._battery_power > 0 or self._is_plugged_in
                '''
                return 'smoke' in air_samples
        self.assertEqual(analyze_class(SmokeDetectorWithBattery), [])

    def TODO_test_cannot_strengthen_inherited_preconditions(self): # TODO: precondition strengthening check
        class PowerHungrySmokeDetector(SmokeDetector):
            _battery_power: int
            def signaling_alarm(self, air_samples: List[str]) -> bool:
                '''
                pre: self._is_plugged_in
                pre: self._battery_power > 0
                '''
                return 'smoke' in air_samples
        self.assertEqual(*check_messages(analyze_class(PowerHungrySmokeDetector),
                                         state=MessageType.PRE_INVALID))

    def test_container_typevar(self) -> None:
        T = TypeVar('T')
        def f(s:Sequence[T]) -> Dict[T, T]:
            ''' post: len(_) == len(s) '''
            return dict(zip(s, s))
        self.assertEqual(*check_fail(f))  # (sequence could contain duplicate items)

    def test_typevar_bounds_fail(self) -> None: 
        T = TypeVar('T')
        def f(x:T) -> int:
            ''' post:True '''
            return x + 1 # type: ignore
        self.assertEqual(*check_exec_err(f))

    def test_typevar_bounds_ok(self) -> None: 
        B = TypeVar('B', bound=int)
        def f(x:B) -> int:
            ''' post:True '''
            return x + 1
        self.assertEqual(*check_ok(f))
        
    def TODO_test_any(self) -> None:
        pass
        
    def test_meeting_class_preconditions(self) -> None:
        def f() -> int:
            '''
            post: _ == -1
            '''
            pokeable = Pokeable(0)
            pokeable.safe_pokeby(-1)
            return pokeable.x
        result = analyze_function(f)
    
    def test_enforced_fn_preconditions(self) -> None:
        def f(x:int) -> bool:
            ''' post: _ == True '''
            return bool(fibb(x)) or True
        self.assertEqual(*check_exec_err(f))

    def test_generic_type_as_data(self) -> None:
        def f(thing: object, detector_kind: Type[SmokeDetector]):
            ''' post: True '''
            if isinstance(thing, detector_kind):
                return thing.is_plugged_in
            return False
        self.assertEqual(*check_ok(f))

class BehaviorsTest(unittest.TestCase):
    def test_syntax_error(self) -> None:
        def f(x:int) -> int:
            ''' pre: x && x '''
        self.assertEqual(*check_messages(analyze_function(f),
                                         state=MessageType.SYNTAX_ERR))

    def test_varargs_fail(self) -> None:
        def f(x:int, *a:str, **kw:bool) -> int:
            ''' post: _ > x '''
            return x + len(a) + (42 if kw else 0)
        self.assertEqual(*check_fail(f))
        
    def test_varargs_ok(self) -> None:
        def f(x:int, *a:str, **kw:bool) -> int:
            ''' post: _ >= x '''
            return x + len(a) + (42 if kw else 0)
        self.assertEqual(*check_unknown(f))
        
    def test_recursive_fn_fail(self) -> None:
        self.assertEqual(*check_fail(fibb))

    def test_recursive_fn_ok(self) -> None:
        self.assertEqual(*check_ok(recursive_example))

    def test_recursive_postcondition_ok(self) -> None:
        def f(x: int) -> int:
            ''' post: _ == f(-x) '''
            return x * x
        self.assertEqual(*check_ok(f))

    def test_repeatable_execution(self) -> None:
        def f(x: int) -> int:
            ''' post: _ >= 1 '''
            return abs(x - 12)
        original_messages = analyze_function(f)
        self.assertEqual(len(original_messages), 1)
        conditions = get_fn_conditions(f)
        replay_analysis = replay(f, original_messages[0].execution_log, conditions)
        original_messages = [replace(original_messages[0], execution_log = None)] # to make comparison
        self.assertEqual(original_messages, replay_analysis.messages)

    def test_recursive_postcondition_enforcement_suspension(self) -> None:
        messages = analyze_class(Measurer)
        self.assertEqual(*check_messages(messages,
                                         state=MessageType.POST_FAIL))

    def test_error_message_has_unmodified_args(self) -> None:
        def f(foo:List[Pokeable]) -> None:
            '''
            pre: len(foo) == 1
            pre: foo[0].x == 10
            post[foo]: foo[0].x == 12
            '''
            foo[0].poke()
        self.assertEqual(*check_messages(analyze_function(f),
                                         state=MessageType.POST_FAIL,
                                         message='false when foo = [Pokeable(10)]'))

    def TODO_test_potential_circular_references(self) -> None: # TODO: List[List] involves no HeapRefs
        def f(foo: List[List], thing: object) -> None: # TODO?: potential aliasing of input argument data?
            '''
            pre: len(foo) == 2
            pre: len(foo[0]) == 1
            pre: len(foo[1]) == 1
            post: len(foo[1]) == 1
            '''
            foo[0].append(object()) # TODO: using 42 yields a z3 sort error
        self.assertEqual(*check_ok(f))
            
        
class ContractedBuiltinsTest(unittest.TestCase):
    
    def TODO_test_print_ok(self) -> None:
        def f(x:int) -> bool:
            '''
            post: _ == True
            '''
            print(x)
            return True
        self.assertEqual(*check_ok(f))

    def test_dispatch(self):
        self.assertEqual(list(contracted_builtins.max.registry.keys()), [object, collections.Iterable])
        
    def test_isinstance(self):
        f = SmtFloat(StateSpace(), float, 'f')
        self.assertFalse(isinstance(f, float))
        self.assertFalse(isinstance(f, int))
        self.assertTrue(contracted_builtins.isinstance(f, float))
        self.assertFalse(contracted_builtins.isinstance(f, int))
    
    def test_max_fail(self) -> None:
        def f(l: List[int]) -> int:
            '''
            post: _ in l
            '''
            return max(l)
        self.assertEqual(*check_exec_err(f)) # TODO: location of precondition failure should be in f()

    def test_max_ok(self) -> None:
        def f(l: List[int]) -> int:
            '''
            pre: bool(l)
            post: _ in l
            '''
            return max(l)
        self.assertEqual(*check_ok(f))

    def test_min_ok(self) -> None:
        def f(l: List[float]) -> float:
            '''
            pre: bool(l)
            post: _ in l
            '''
            return min(l)
        self.assertEqual(*check_ok(f))

    # TODO: min test  (this breaks b/c enforcement wrapper messes with itself)

    def test_contracted_other_packages(self) -> None:
        # TODO make args be real dates and more preconditions into wrapper
        import datetime
        def f(y:int, m:int, d:int, num_days: int) -> datetime.date:
            '''
            pre: 2000 <= y <= 2020
            pre: 1 <= m <= 12
            pre: 1 <= d <= 28
            pre: num_days == -10
            pre: datetime.date(y,m,d)
            post: _.year >= y
            '''
            return datetime.date(y,m,d) + datetime.timedelta(days = int(num_days))
        self.assertEqual(*check_fail(f))
        
    
class CallableTest(unittest.TestCase):
    def test_symbolic_zero_arg_callable(self) -> None:
        def f(size:int, initializer:Callable[[], int]) -> Tuple[int, ...]:
            '''
            pre: size >= 1
            post: _[0] != 707
            '''
            return tuple(initializer() for _ in range(size))
        self.assertEqual(*check_fail(f))

    def test_symbolic_one_arg_callable(self) -> None:
        def f(size:int, mapfn:Callable[[int], int]) -> Tuple[int, ...]:
            '''
            pre: size >= 1
            post: _[0] != 707
            '''
            return tuple(mapfn(i) for i in range(size))
        self.assertEqual(*check_fail(f))

    def test_symbolic_two_arg_callable(self) -> None:
        def f(i:int, callable:Callable[[int, int], int]) -> int:
            ''' post: _ != i '''
            return callable(i, i)
        self.assertEqual(*check_fail(f))


class LargeExamplesTest(unittest.TestCase):

    def test_arith(self) -> None:
        messages = analyze_module(crosshair.examples.arith, AnalysisOptions())
        self.assertEqual(*check_messages(messages,
                                         state=MessageType.EXEC_ERR,
                                         line=39))

    def TODO_test_tic_tac_toe(self) -> None:
        self.assertEqual(
            analyze_class(crosshair.examples.tic_tac_toe.Board),
            [])

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
    #suite = unittest.TestLoader().loadTestsFromTestCase(CoreTest)
    #unittest.TextTestRunner(verbosity=2).run(suite)
