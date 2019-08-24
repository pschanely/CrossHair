from typing import Collection, Iterable, List, Tuple, TypeVar


T = TypeVar('T')
U = TypeVar('U')

def duplicate_list(a:List[T]) -> List[T]:
    '''
    #post: len(return) == 2 * len(a)
    #post: return[:len(a)] == a
    #post: return[-len(a):] == a
    '''
    return a + a

def compute_grade(homework_scores:List[float], exam_scores:List[float]) -> float:
    '''
    pre: homework_scores or exam_scores
    pre: bool(all(0 <= s <= 1.0 for s in (homework_scores + exam_scores)))
    post: return < 1.0
    '''
    # make exams matter more by counting them twice:
    all_scores = homework_scores + exam_scores + exam_scores
    return sum(all_scores) / len(all_scores)

def zip_exact(a:Iterable[T], b:Iterable[U]) -> Iterable[Tuple[T, U]]:
    '''
    #pre: len(a) == len(b)
    #post: len(return) == len(a) == len(b)
    '''
    return zip(a, b)

