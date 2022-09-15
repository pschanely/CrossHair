import statistics
from typing import Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def average(numbers: List[float]) -> float:
    """
    pre: len(numbers) > 0
    post: min(numbers) <= __return__ <= max(numbers)
    """
    return sum(numbers) / len(numbers)


def duplicate_list(a: List[T]) -> List[T]:
    """
    post: len(__return__) == 2 * len(a)
    post: __return__[:len(a)] == a
    post: __return__[-len(a):] == a
    """
    return a + a


def compute_grade(homework_scores: List[float], exam_scores: List[float]) -> float:
    """
    pre: homework_scores or exam_scores
    pre: all(0 <= s <= 1.0 for s in homework_scores + exam_scores)
    post: 0 <= __return__ <= 1.0
    """
    # make exams matter more by counting them twice:
    all_scores = homework_scores + exam_scores + exam_scores
    return sum(all_scores) / len(all_scores)


def make_csv_line(objects: Sequence) -> str:
    """
    pre: len(objects) > 0
    pre: all(',' not in str(o) for o in objects)
    post: __return__.split(',') == list(map(str, objects))
    """
    return ",".join(map(str, objects))


def csv_first_column(lines: List[str]) -> List[str]:
    """
    pre: all(',' in line for line in lines)
    post: __return__ == [line.split(',')[0] for line in lines]
    """
    return [line[: line.index(",")] for line in lines]


def zip_exact(a: Iterable[T], b: Iterable[U]) -> List[Tuple[T, U]]:
    """
    pre: len(a) == len(b)
    post: len(__return__) == len(a) == len(b)
    """
    return list(zip(a, b))


def zipped_pairs(x: List[T]) -> List[Tuple[T, T]]:
    """
    post: len(__return__) == max(0, len(x) - 1)
    """
    return zip_exact(x[:-1], x[1:])


def even_fibb(n: int) -> List[int]:
    """
    Return a list of the first N even fibbonacci numbers.

    >>> even_fibb(2)
    [2, 8]

    pre: n >= 0
    post: len(__return__) == n
    """
    prev = 1
    cur = 1
    result = []
    while n > 0:
        prev, cur = cur, prev + cur
        if cur % 2 == 0:
            result.append(cur)
            n -= 1
    return result


def remove_outliers(numbers: List[float], num_deviations: float = 3):
    """
    >>> remove_outliers([0, 1, 2, 3, 4, 5, 50, 6, 7, 8, 9], num_deviations=1)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    post: len(_) <= len(numbers)
    post: not numbers or max(_) <= max(numbers)
    post: not numbers or min(_) >= min(numbers)
    post: all(x in numbers for x in _)
    """
    if len(numbers) < 2:
        return numbers
    avg = statistics.mean(numbers)
    allowed_range = statistics.stdev(numbers) * num_deviations
    min_val, max_val = avg - allowed_range, avg + allowed_range
    return [num for num in numbers if min_val <= num <= max_val]
