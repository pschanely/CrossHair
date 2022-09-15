import statistics
from typing import Iterable, List, Sequence, Tuple, TypeVar

from icontract import ensure, require

T = TypeVar("T")
U = TypeVar("U")


@require(lambda numbers: len(numbers) > 0)
@ensure(lambda numbers, result: min(numbers) <= result <= max(numbers))
def average(numbers: List[float]) -> float:
    return sum(numbers) / len(numbers)


@ensure(lambda a, result: len(result) == 2 * len(a))
@ensure(lambda a, result: result[: len(a)] == a)
@ensure(lambda a, result: result[-len(a) :] == a)
def duplicate_list(a: List[T]) -> List[T]:
    return a + a


@require(lambda homework_scores, exam_scores: homework_scores or exam_scores)
@require(
    lambda homework_scores, exam_scores: all(
        0 <= s <= 1.0 for s in homework_scores + exam_scores
    )
)
@ensure(lambda result: 0 <= result <= 1.0)
def compute_grade(homework_scores: List[float], exam_scores: List[float]) -> float:
    # Make exams matter more by counting them twice:
    all_scores = homework_scores + exam_scores + exam_scores
    return sum(all_scores) / len(all_scores)


@require(lambda objects: len(objects) > 0)
@require(lambda objects: all("," not in str(o) for o in objects))
@ensure(lambda objects, result: result.split(",") == list(map(str, objects)))
def make_csv_line(objects: Sequence) -> str:
    return ",".join(map(str, objects))


@require(lambda lines: all("," in line for line in lines))
@ensure(lambda lines, result: result == [line.split(",")[0] for line in lines])
def csv_first_column(lines: List[str]) -> List[str]:
    return [line[: line.index(",")] for line in lines]


@require(lambda a, b: len(a) == len(b))
@ensure(lambda a, b, result: len(result) == len(a) == len(b))
def zip_exact(a: Iterable[T], b: Iterable[U]) -> List[Tuple[T, U]]:
    return list(zip(a, b))


@ensure(lambda x, result: len(result) == max(0, len(x) - 1))
def zipped_pairs(x: List[T]) -> List[Tuple[T, T]]:
    return zip_exact(x[:-1], x[1:])


@require(lambda n: n >= 0)
@ensure(lambda n, result: len(result) == n)
def even_fibb(n: int) -> List[int]:
    """
    Return a list of the first N even fibbonacci numbers.

    >>> even_fibb(2)
    [2, 8]
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


@ensure(lambda numbers, result: len(result) <= len(numbers))
@ensure(lambda numbers, result: not numbers or max(result) <= max(numbers))
@ensure(lambda numbers, result: not numbers or min(result) >= min(numbers))
@ensure(lambda numbers, result: all(x in numbers for x in result))
def remove_outliers(numbers: List[float], num_deviations: float = 3):
    """
    >>> remove_outliers([0, 1, 2, 3, 4, 5, 50, 6, 7, 8, 9], num_deviations=1)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if len(numbers) < 2:
        return numbers
    avg = statistics.mean(numbers)
    allowed_range = statistics.stdev(numbers) * num_deviations
    min_val, max_val = avg - allowed_range, avg + allowed_range
    return [num for num in numbers if min_val <= num <= max_val]
