from crosshair.test_util import flexible_equal


def test_flexible_equal():
    assert float("nan") != float("nan")
    assert flexible_equal(float("nan"), float("nan"))
    assert flexible_equal((42, float("nan")), (42, float("nan")))
    assert not flexible_equal([float("nan"), 11], [float("nan"), 22])

    def gen():
        yield 11
        yield 22

    assert flexible_equal(gen(), iter([11, 22]))
    assert not flexible_equal(gen(), iter([11, 22, 33]))
    assert not flexible_equal(gen(), iter([11]))

    ordered_set_1 = {10_000, 20_000} | {30_000}
    ordered_set_2 = {30_000, 20_000} | {10_000}
    assert list(ordered_set_1) != list(ordered_set_2)  # (different orderings)
    assert flexible_equal(ordered_set_1, ordered_set_2)

    ordered_dict_1 = {1: 2, 3: 4}
    ordered_dict_2 = {3: 4, 1: 2}
    assert list(ordered_dict_1.items()) != list(ordered_dict_2.items())
    assert flexible_equal(ordered_dict_1, ordered_dict_2)
