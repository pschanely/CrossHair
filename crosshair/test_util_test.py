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
