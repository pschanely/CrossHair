from typing import *


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
    #pre: homework_scores or exam_scores
    #pre: all(0 <= s <= 1.0 for s in homework_scores + exam_scores)
    #post: 0 <= return <= 1.0
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

def list_to_dict(s:Sequence[T]) -> Dict[T, T]:
    '''
    #post: len(return) == len(s)
    '''
    return dict(zip(s, s))

def make_csv_line(objects: Sequence[str]) -> str:
    '''
    #pre: objects
    #post: return.split(',') == list(map(str, objects))
    '''
    return ','.join(map(str, objects))

## TODO - contracted modules
#import datetime
#def add_days(dt: datetime.date, num_days: int) -> datetime.date:
#    '''
#    post: return > dt
#    '''
#    return dt + datetime.timedelta(days = num_days)
