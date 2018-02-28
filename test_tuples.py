from crosshair import *

def tuples1() -> istrue: return (1,2) == (*(1,), 2)
def tuples2() -> istrue: return (1, 2) == (1, *(2,))
def tuples3() -> istrue: return (1,2,2) == (1, *(2,), 2)
def tuples4() -> istrue: return istuple((1, *(2,3)))

def len1() -> istrue: return len(()) == 0
def len2() -> istrue: return len((1,2)) == 2
def len3() -> istrue: return isdefined(len((1,3)))
def len4() -> istrue: return len((1, *(2,))) == 2
def len5() -> istrue: return len((1,3)) == 2
def len6() -> istrue: return len((1,*(2,3,4),5)) == 5
def len7() -> istrue: return len((1,)+(2,)) == 2
def len8(t:istuple) -> istrue: return len(t) < len((*t, 1))


def all_true_on_empty() -> istrue: return all(())
def all_on_literals1() -> istrue: return all((True,))
def all_on_literals2() -> istrue: return all((True, True))
def all_ignore_true_values1(t:istuple) -> istrue: return implies(all(t), all((*t, True)))


## Induction required here:
#def all_ignore_true_values2(t:istuple) -> istrue: return implies(all(t), all((True, *t)))


