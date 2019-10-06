from typing import *


T = TypeVar('T')
U = TypeVar('U')

def duplicate_list(a:List[T]) -> List[T]:
    '''
    #post: len(__return__) == 2 * len(a)
    #post: __return__[:len(a)] == a
    #post: __return__[-len(a):] == a
    '''
    return a + a

def compute_grade(homework_scores:List[float], exam_scores:List[float]) -> float:
    '''
    #pre: homework_scores or exam_scores
    #pre: all(0 <= s <= 1.0 for s in homework_scores + exam_scores)
    #post: 0 <= __return__ <= 1.0
    '''
    # make exams matter more by counting them twice:
    all_scores = homework_scores + exam_scores + exam_scores
    return sum(all_scores) / len(all_scores)

def zip_exact(a:Iterable[T], b:Iterable[U]) -> Iterable[Tuple[T, U]]:
    '''
    #pre: len(a) == len(b)
    #post: len(__return__) == len(a) == len(b)
    '''
    return zip(a, b)

def list_to_dict(s:Sequence[T]) -> Dict[T, T]:
    '''
    #post: len(__return___) == len(s)
    '''
    return dict(zip(s, s))

def make_csv_line(objects: Sequence[str]) -> str:
    '''
    #pre: objects
    #post: __return__.split(',') == list(map(str, objects))
    '''
    return ','.join(map(str, objects))

def csv_first_column(lines:List[str]) -> List[str]:
    '''
    #pre: all(',' in line for line in lines)
    #post: __return__ == [line.split(',')[0] for line in lines]
    '''
    return [line[:line.index(',')] for line in lines]

## TODO - contracted modules
#import datetime
#def add_days(dt: datetime.date, num_days: int) -> datetime.date:
#    '''
#    post: __return__ > dt
#    '''
#    return dt + datetime.timedelta(days = num_days)
