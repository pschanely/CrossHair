import unittest

from core import *





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
        self.x += 1
    def wild_pokeby(self, amount:int) -> None:
        self.x += amount
    def safe_pokeby(self, amount:int) -> None:
        '''
        pre: amount >= 0
        '''
        self.x += amount

#
# End fixed line number area.
#




def check_fail(fn):
    return ([m.state for m in analyze(fn)], ['post_fail'])

def check_exec_err(fn):
    return ([m.state for m in analyze(fn)], ['exec_err'])

def check_post_err(fn):
    return ([m.state for m in analyze(fn)], ['post_err'])

def check_unknown(fn):
    return ([m.state for m in analyze(fn)], ['cannot_confirm'])

def check_ok(fn):
    return (analyze(fn), [])



def fibb(x:int) -> int:
    '''
    pre: x>=0
    post: return < 100
    '''
    if x <= 2:
        return 1
    return fibb(x-1) + fibb(x-2)

def recursive_example(x:int) -> bool:
    '''
    pre: x >= 0
    post: return == True
    '''
    if x == 0:
        return True
    else:
        return recursive_example(x - 1)

class CoreTest(unittest.TestCase):

    def test_simple_bool_with_fail(self) -> None:
        def f(a:bool, b:bool) -> bool:
            '''
            post: return == a
            '''
            return True if a else b
        self.assertEqual(*check_fail(f))

    def test_simple_bool_ok(self) -> None:
        def f(a:bool, b:bool) -> bool:
            '''
            post: return == a or b
            '''
            return True if a else b
        self.assertEqual(*check_ok(f))
        
    def test_bool_ors_fail(self) -> None:
        def f(a:bool, b:bool, c:bool, d:bool) -> bool:
            '''
            post: return == (a ^ b) or (c ^ d)
            '''
            return a or b or c or d
        self.assertEqual(*check_fail(f))
        
    def test_bool_ors(self) -> None:
        def f(a:bool, b:bool, c:bool, d:bool) -> bool:
            '''
            pre: (not a) and (not d)
            post: return == (a ^ b) or (c ^ d)
            '''
            return a or b or c or d
        self.assertEqual(*check_ok(f))
        
    def test_numeric_promotions(self) -> None:
        def f(b:bool, i:int) -> Tuple[int, float, float]:
            '''
            #post: 100 <= return[0] <= 101
            #post: 3.14 <= return[1] <= 4.14
            post: isinstance(return[2], float)
            '''
            return ((b + 100), (b + 3.14), (i + 3.14))
        self.assertEqual(*check_ok(f))

    def test_int_reverse_operators(self) -> None:
        def f(i:int) -> float:
            '''
            pre: i != 0
            post: return > 0
            '''
            return (1 + i) + (1 - i) + (1 / i)
        self.assertEqual(*check_ok(f))

    def test_int_div_fail(self) -> None:
        def f(a:int, b:int) -> int:
            '''
            post: a <= return <= b
            '''
            return (a + b) // 2
        self.assertEqual(*check_fail(f))

    def test_int_div_ok(self) -> None:
        def f(a:int, b:int) -> int:
            '''
            pre: a < b
            post: a <= return <= b
            '''
            return (a + b) // 2
        self.assertEqual(*check_ok(f))

    def test_true_div_fail(self) -> None:
        def f(a:int, b:int) -> float:
            '''
            pre: a != 0 and b != 0
            post: return >= 1.0
            '''
            return (a + b) / b
        self.assertEqual(*check_fail(f))
    
    def test_true_div_ok(self) -> None:
        def f(a:int, b:int) -> float:
            '''
            pre: a >= 0 and b > 0
            post: return >= 1.0
            '''
            return (a + b) / b
        self.assertEqual(*check_ok(f))

    def test_string_multiply_fail(self) -> None:
        def f(a:str) -> str:
            '''
            post: len(return) == len(a) * 3
            '''
            return 3 * a
        self.assertEqual(*check_ok(f))

    def test_string_multiply_ok(self) -> None:
        def f(a:str) -> str:
            '''
            post: len(return) == len(a) * 5
            '''
            return a * 3 + 2 * a
        self.assertEqual(*check_ok(f))
    
    def test_string_prefixing(self) -> None:
        def f(a:str, indent:bool) -> str:
            '''
            post: len(return) == len(a) + indent
            '''
            return ('  ' if indent else '') + a
        self.assertEqual(*check_fail(f))
        def f(a:str, indent:bool) -> str:
            '''
            post: len(return) == len(a) + (2 if indent else 0)
            '''
            return ('  ' if indent else '') + a
        self.assertEqual(*check_ok(f))

    def test_string_negative_index_slicing(self) -> None:
        def f(s:str) -> Tuple[str, str]:
            '''
            post: sum(map(len,return)) == len(s) - 1
            '''
            idx = s.find(':')
            return (s[:idx], s[idx+1:])
        self.assertEqual(*check_fail(f))  # (fails when idx == -1)
        
    def test_tuple_range_intersection_fail(self) -> None:
        def f(a:Tuple[int, int], b:Tuple[int, int]) -> Optional[Tuple[int, int]]:
            '''
            pre: a[0] < a[1] and b[0] < b[1]
            post: return[0] <= return[1]
            '''
            return (max(a[0], b[0]), min(a[1], b[1]))
        self.assertEqual(*check_fail(f))
    
    def test_tuple_range_intersection_ok(self) -> None:
        def f(a:Tuple[int, int], b:Tuple[int, int]) -> Optional[Tuple[int, int]]:
            '''
            pre: a[0] < a[1] and b[0] < b[1]
            post: return is None or return[0] <= return[1]
            '''
            if a[1] > b[0] and a[0] < b[1]: # (if the ranges overlap)
                return (max(a[0], b[0]), min(a[1], b[1]))
            else:
                return None
        self.assertEqual(*check_ok(f))

    def test_list_containment(self) -> None:
        def f(a:int, b:List[int]) -> bool:
            '''
            post: return == (a in b[:5])
            '''
            return a in b
        self.assertEqual(*check_fail(f))
        def f(a:int, b:List[int]) -> bool:
            '''
            pre: 1 == len(b)
            post: return == (a == b[0])
            '''
            return a in b
        self.assertEqual(*check_ok(f))
        
    def test_list_doubling_fail(self) -> None:
        def f(a:List[int]) -> List[int]:
            '''
            post: len(return) > len(a)
            '''
            return a + a
        self.assertEqual(*check_fail(f))

    def test_list_doubling_ok(self) -> None:
        def f(a:List[int]) -> List[int]:
            '''
            post: len(return) > len(a) or not a
            '''
            return a + a
        self.assertEqual(*check_ok(f))

    def test_list_range_fail(self) -> None:
        def f(l:List[int]) -> List[int]:
            '''
            pre: len(l) == 3
            post: len(return) > len(l)
            '''
            n:List[int] = [] 
            for i in range(len(l)):
                n.append(l[i] + 1)
            return n
        self.assertEqual(*check_fail(f))
    
    def test_list_range_ok(self) -> None:
        def f(l:List[int]) -> List[int]:
            '''
            pre: l and len(l) < 10  # (max is to cap runtime)
            post: return[0] == l[0] + 1
            '''
            n:List[int] = [] 
            for i in range(len(l)):
                n.append(l[i] + 1)
            return n
        self.assertEqual(*check_ok(f))

    def test_list_extend_literal_unknown(self) -> None:
        def f(l:List[int]) -> List[int]:
            '''
            post: return[:2] == [1, 2]
            '''
            r = [1,2,3]
            r.extend(l)
            return r
        self.assertEqual(*check_unknown(f))
            
    def test_list_index_error(self) -> None:
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
            post: return > 0
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
            post: return >= 0
            '''
            total = 0
            for i in l:
                total += len(i)
            return total
        self.assertEqual(*check_ok(f))
        
    # TODO test list slice outside range
    
    #
    # dictionaries
    #
    
    def test_dict_basic_fail(self) -> None:
        def f(a:Dict[int, str], k:int, v:str) -> None:
            '''
            post: a[k] == ""
            '''
            a[k] = v
        self.assertEqual(*check_fail(f))

    def test_dict_basic_ok(self) -> None:
        def f(a:Dict[int, str], k:int, v:str) -> None:
            '''
            post: a[k] == v
            '''
            a[k] = v
        self.assertEqual(*check_ok(f))

    def test_dict_empty_bool(self) -> None:
        def f(a:Dict[int, str]) -> bool:
            '''
            post: return == True
            '''
            a[0] = 'zero'
            return bool(a)
        self.assertEqual(*check_ok(f))

    def test_dict_iter_fail(self) -> None:
        def f(a:Dict[int, str]) -> List[int]:
            '''
            post: 5 in return
            '''
            a[10] = 'ten'
            return list(a)
        self.assertEqual(*check_fail(f))

    def test_dict_iter_ok(self) -> None:
        def f(a:Dict[int, str]) -> List[int]:
            '''
            pre: len(a) < 3
            post: 5 in return
            '''
            a[10] = 'ten'
            return list(a)
        # TODO: passes, but only because precondition explodes
        self.assertEqual(*check_ok(f))

    # TODO raise warning when function cannot complete successfully
        
    #
    # custom objects
    #
    
    def test_obj_member_fail(self) -> None:
        def f(foo:Pokeable) -> int:
            '''
            post: return < 5
            '''
            foo.poke()
            foo.poke()
            return foo.x
        self.assertEqual(*check_fail(f))
        
    def test_obj_member_ok(self) -> None:
        def f(foo:Pokeable) -> int:
            '''
            pre: foo.x >= 0
            post: foo.x >= 2
            '''
            foo.poke()
            foo.poke()
            return foo.x
        self.assertEqual(*check_ok(f))

    def test_example_second_largest(self) -> None:
        def second_largest(items: List[int]) -> int:
            '''
            pre: 2 <= len(items) <= 3  # (max is to cap runtime)
            post: return == sorted(items)[-2]
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
        self.assertEqual(
            analyze_class(Pokeable),
            [AnalysisMessage(
                state='post_fail',
                message='false when calling wild_pokeby with self.x = 0 and amount = -1',
                filename='crosshair/core_test.py',
                line=16,
                column=0)
             ])

    def test_typevar(self) -> None:
        T = TypeVar('T')
        class MaybePair(Generic[T]):
            '''
            inv: (self.left is None) == (self.right is None)
            '''
            def __init__(self, left:Optional[T], right:Optional[T]):
                if (left is None) ^ (right is None):
                    raise ValueError('one side must be given if the other is')
                self.left, self.right = left, right
        messages = analyze_class(MaybePair)
        self.assertEqual(len(messages), 1)
        message = messages[0].message
        self.assertRegex(message, r'one side must be given if the other is')
        self.assertRegex(message, r' type\(left\) ')
        self.assertRegex(message, r' type\(right\) ')
        self.assertRegex(message, r'\bNoneType\b')
        self.assertRegex(message, r'\bT\b')

    def XXX_test_varargs(self) -> None:
        def f(x:int, *a:str, **kw:bool) -> int:
            '''
            post: return >= x
            '''
            print('a', str(a))
            return x + len(a) + (42 if kw else 0)
        self.assertEqual(*check_ok(f))
        
    def XXX_test_any(self) -> None:
        pass
        
    def test_meeting_class_preconditions(self) -> None:
        def f() -> int:
            '''
            post: return == 0
            '''
            pokeable = Pokeable()
            pokeable.safe_pokeby(-1)
            return pokeable.x
        self.assertEqual(
            analyze(f),
            [AnalysisMessage(
                state='exec_err',
                message='PreconditionFailed: Precondition failed at crosshair/core_test.py:25 for any input',
                filename='/Users/pschanely/Dropbox/wf/CrossHair/crosshair/enforce.py',
                line=24,
                column=0)
            ])
    
    def test_meeting_fn_preconditions(self) -> None:
        def f(x:int) -> bool:
            '''
            post: return == True
            '''
            return bool(fibb(x)) or True
        self.assertEqual(*check_exec_err(f))

    def test_recursive_fn_fail(self) -> None:
        self.assertEqual(*check_fail(fibb))

    def test_recursive_fn_ok(self) -> None:
        self.assertEqual(*check_ok(recursive_example))
        
if __name__ == '__main__':
    unittest.main()
