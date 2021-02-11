from crosshair.options import DEFAULT_OPTIONS


def test_AnalysisOptions_split_limits() -> None:
    options = DEFAULT_OPTIONS.overlay(per_path_timeout=10.0, max_iterations=16)
    part1, part2 = options.split_limits(0.1)
    assert part1.per_path_timeout == 1.0
    assert part2.per_path_timeout == 9.0
    assert part1.max_iterations == 2
    assert part2.max_iterations == 14
