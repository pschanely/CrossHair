import pytest

from crosshair.simplestructs import (
    LazySetCombination,
    SequenceConcatenation,
    ShellMutableMap,
    ShellMutableSequence,
    ShellMutableSet,
    SimpleDict,
    SingletonSet,
    SliceView,
    cut_slice,
    operator,
)
from crosshair.test_util import summarize_execution


def test_ShellMutableMap() -> None:
    m = ShellMutableMap({4: 4, 5: 5, 6: 6})
    m[3] = 3
    m[4] = None
    del m[5]
    assert len(m) == 3
    assert list(m) == [6, 3, 4]
    assert m[4] is None
    with pytest.raises(KeyError):
        m[5]
    assert m == {3: 3, 4: None, 6: 6}


def test_ShellMutableMap_popitem_ordering() -> None:
    assert {"c": "d", "a": "b"}.popitem() == ("a", "b")
    assert SimpleDict([("c", "d"), ("a", "b")]).popitem() == ("a", "b")


def test_ShellMutableMap_poo() -> None:
    m = ShellMutableMap({2: 0})
    assert 0 == m.setdefault(2.0, {True: "0"})


def test_SequenceConcatenation_comparison() -> None:
    compound = SequenceConcatenation((11, 22), (33, 44))
    assert compound == (11, 22, 33, 44)
    assert compound < (22, 33, 44)
    assert compound >= (11, 22, 33)  # type: ignore


def test_SliceView_comparison() -> None:
    sliced = SliceView((00, 11, 22, 33, 44, 55), 1, 5)
    assert sliced == (11, 22, 33, 44)
    assert sliced < (22, 33, 44)
    assert sliced >= (11, 22, 33)  # type: ignore


def test_slice_view() -> None:
    nums = ["0", "1", "2", "3", "4", "5"]
    ctr = 0
    for start in [0, 1, 2, 3, 4, 5, 6]:
        for stop in range(start, 7):
            view = SliceView(nums, start, stop)
            concrete = nums[start:stop]
            assert (
                list(view) == concrete
            ), f"{ctr}: {start}:{stop}: {view} vs {concrete}"
            if stop - start > 0:
                assert view[0] == nums[start]
            ctr += 1


def test_LazySetCombination_xor() -> None:
    a = {2, 4, 6}
    b = {4, 5, 6, 7}
    s = LazySetCombination(operator.xor, a, b)
    assert s == {2, 5, 7}
    assert 4 not in s
    assert 5 in s


def test_ShellMutableSequence_slice_assignment() -> None:
    ls = ["0", "1", "2", "3"]
    shell = ShellMutableSequence(ls)
    assert shell == shell
    assert shell == ls
    shell[1:3] = ["1", "1.5", "2"]
    assert shell == ["0", "1", "1.5", "2", "3"]
    assert shell == shell


def test_ShellMutableSequence_assignment_negative_index() -> None:
    ls = ShellMutableSequence(["a", "a"])
    ls[-1] = "b"
    assert ls == ["a", "b"]


def test_ShellMutableSequence_assignment_bad_index() -> None:
    with pytest.raises(TypeError):
        ShellMutableSequence([])["whoa"] = 3


def test_ShellMutableSequence_assignment_out_of_bounds() -> None:
    with pytest.raises(IndexError):
        ShellMutableSequence(["b", "c"])[-3] = "a"


def test_ShellMutableSequence_sort_invalid_args() -> None:
    s = ShellMutableSequence(SequenceConcatenation([], []))
    with pytest.raises(TypeError):
        s.sort(reverse="badvalue")


def test_ShellMutableSequence_assignment_doesnt_pollute() -> None:
    a = ShellMutableSequence([1, 2])
    b = a + ShellMutableSequence([3, 4])
    assert b == [1, 2, 3, 4]
    a.append(3)
    assert b == [1, 2, 3, 4]


def test_ShellMutableSequence_assignment_doesnt_get_polluted() -> None:
    a = [1, 2]
    b = a + ShellMutableSequence([3, 4])
    assert b == [1, 2, 3, 4]
    a.append(3)
    assert b == [1, 2, 3, 4]


def test_SequenceConcatenation_operators() -> None:
    s = SequenceConcatenation([4], [6]) + [8]
    assert s == [4, 6, 8]
    assert type(s) == SequenceConcatenation
    s = [2] + SequenceConcatenation([4], [6])
    assert s == [2, 4, 6]
    assert type(s) == SequenceConcatenation


def test_SingletonSet_arg_types() -> None:
    s = SingletonSet(42)
    with pytest.raises(TypeError):
        s | [3, 5]
    with pytest.raises(TypeError):
        [3, 5] ^ s  # type: ignore
    with pytest.raises(TypeError):
        s - [3, 5]


def test_ShellMutableSet_members() -> None:
    shell = ShellMutableSet([6, 4, 8, 2])
    assert shell == shell
    assert shell == {2, 4, 6, 8}
    shell.pop()
    assert list(shell) == [4, 8, 2]
    shell.add(0)
    assert list(shell) == [4, 8, 2, 0]
    shell.clear()
    assert list(shell) == []


def test_ShellMutableSet_comparisons() -> None:
    # Truths
    assert ShellMutableSet([2, 4]) >= {2, 4}
    assert ShellMutableSet([2, 4]) < {2, 3, 4}
    assert ShellMutableSet([2, 4]) == {4, 2}
    assert ShellMutableSet([2, 4]) != {}
    assert ShellMutableSet([2, 4]).isdisjoint({1, 3, 5})
    # Falsehoods
    assert not (ShellMutableSet([2, 4]) > {2, 4})
    assert not (ShellMutableSet([2, 4]) <= {4, 5})
    assert not (ShellMutableSet([2, 4]).isdisjoint({1, 4, 5}))


def test_ShellMutableSet_operators() -> None:
    shell = ShellMutableSet([2, 4])
    shell = shell | {6}
    assert len(shell) == 3
    assert list(shell) == [2, 4, 6]
    shell = shell & {2, 6, 8}
    assert list(shell) == [2, 6]
    shell = shell ^ {0, 2, 4, 6}
    assert list(shell) == [0, 4]
    shell = shell - {0, 1}
    assert list(shell) == [4]
    shell = {3, 4, 5} - shell  # type: ignore
    assert isinstance(shell, ShellMutableSet)
    assert list(shell) == [3, 5]


def test_ShellMutableSet_mutating_operators() -> None:
    shell = ShellMutableSet([2, 4])
    shell |= {6}
    assert list(shell) == [2, 4, 6]
    shell &= {2, 6, 8}
    assert list(shell) == [2, 6]
    shell ^= {0, 2, 4, 6}
    assert list(shell) == [0, 4]
    shell -= {0, 1}
    assert list(shell) == [4]


def test_ShellMutableSet_errors() -> None:
    with pytest.raises(KeyError):
        ShellMutableSet([2]).remove(3)
    with pytest.raises(KeyError):
        ShellMutableSet([]).pop()
    with pytest.raises(TypeError):
        ShellMutableSet(4)


@pytest.mark.parametrize("start", [-3, -1, 0, 1, 3, None])
@pytest.mark.parametrize("stop", [-2, -1, 0, 1, 2, None])
def test_slice_view_slice(start, stop) -> None:
    allnums = ["0", "1", "2", "3"]
    innernums = SliceView(allnums, 1, 3)
    concrete = ["1", "2"][start:stop]
    assert list(innernums[start:stop]) == concrete


@pytest.mark.parametrize("cut", [-2, 1, 3, 7])
@pytest.mark.parametrize("step", [-2, -1, 1, 2, 3])
@pytest.mark.parametrize("stop", [0, 2, 4, 6])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 5, 6])
def test_cut_slice(start, stop, step, cut):
    left, right = cut_slice(start, stop, step, cut)
    litems = list(range(left.start, left.stop, left.step))
    ritems = list(range(right.start, right.stop, right.step))
    expected = list(range(start, stop, step))
    assert expected == litems + ritems, f"{litems}+{ritems}, expected {expected}"


@pytest.mark.parametrize("idx", [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
def test_SequenceConcatenation_indexed(idx) -> None:
    c1 = SequenceConcatenation(SequenceConcatenation((11, 22), ()), (33,))
    # c1 = SequenceConcatenation((11, 22), (33,))
    c2 = [11, 22, 33]
    r1 = summarize_execution(lambda x: c1[x], (idx,), detach_path=False)
    expected = summarize_execution(lambda x: c2[x], (idx,), detach_path=False)
    assert r1 == expected


@pytest.mark.parametrize("step", [None, -1, 1, 2, -2, 3, -3])
@pytest.mark.parametrize("stop", [None, 0, 2, 4, 6, -2, -4, -6])
@pytest.mark.parametrize("start", [None, 0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
def test_SequenceConcatenation_sliced(start, stop, step) -> None:
    c1 = SequenceConcatenation((11, 22, 33), (44, 55))
    c2 = [11, 22, 33, 44, 55]
    s = slice(start, stop, step)
    r1 = list(c1[s])
    expected = c2[s]
    assert r1 == expected
