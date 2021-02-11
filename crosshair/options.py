import argparse
import collections
from dataclasses import dataclass
from dataclasses import replace
import enum
import sys
from typing import Mapping, Optional, Sequence, Tuple


class AnalysisKind(enum.Enum):
    PEP316 = "PEP316"
    icontract = "icontract"
    asserts = "asserts"
    # hypothesis = "hypothesis"
    def __str__(self):
        return self.value


@dataclass
class AnalysisOptionSet:
    """
    Encodes some set of partially-specified options.

    This class is used while parsing options from various places.
    It is very similar to `AnalysisOptions` (which is used during execution) but allows
    None values everywhere so that options can correctly override each other.
    """

    per_condition_timeout: Optional[float] = None
    per_path_timeout: Optional[float] = None
    timeout: Optional[float] = None
    max_iterations: Optional[int] = None
    report_all: Optional[bool] = None
    analysis_kind: Optional[Sequence[AnalysisKind]] = None

    def overlay(self, overrides: "AnalysisOptionSet") -> "AnalysisOptionSet":
        kw = {k: v for (k, v) in overrides.__dict__.items() if v is not None}
        return replace(self, **kw)


def option_set_from_dict(source: Mapping[str, object]) -> AnalysisOptionSet:
    options = AnalysisOptionSet()
    for optname in (
        "per_path_timeout",
        "per_condition_timeout",
        "report_all",
        "analysis_kind",
    ):
        arg_val = source.get(optname, None)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options


@dataclass
class AnalysisOptions:
    """Encodes the options for use while running CrossHair."""

    per_condition_timeout: float
    per_path_timeout: float
    timeout: float
    max_iterations: int
    report_all: bool
    analysis_kind: Sequence[AnalysisKind]

    # Transient members (not user-configurable):
    deadline: float = float("NaN")
    stats: Optional[collections.Counter] = None

    def overlay(self, overrides: AnalysisOptionSet = None, **kw) -> "AnalysisOptions":
        if overrides is not None:
            assert not kw
            kw = overrides.__dict__
        kw = {k: v for (k, v) in kw.items() if v is not None}
        ret = replace(self, **kw)
        assert type(ret) is AnalysisOptions
        return ret

    def split_limits(
        self, priority: float
    ) -> Tuple["AnalysisOptions", "AnalysisOptions"]:
        """
        Divide resource allotments into two.

        Namely, the resource allotments (timeouts, iteration caps) are split
        into allotments for two stages of analysis.

        pre: 0.0 <= priority <= 1.0
        post: _[0].max_iterations + _[1].max_iterations == self.max_iterations
        """
        options1 = replace(
            self,
            per_condition_timeout=self.per_condition_timeout * priority,
            per_path_timeout=self.per_path_timeout * priority,
            max_iterations=round(self.max_iterations * priority),
        )
        inv_priority = 1.0 - priority
        options2 = replace(
            self,
            per_condition_timeout=self.per_condition_timeout * inv_priority,
            per_path_timeout=self.per_path_timeout * inv_priority,
            max_iterations=self.max_iterations - options1.max_iterations,
        )
        return (options1, options2)

    def incr(self, key: str):
        if self.stats is not None:
            self.stats[key] += 1


DEFAULT_OPTIONS = AnalysisOptions(
    per_condition_timeout=1.5,
    per_path_timeout=0.75,
    timeout=float("inf"),
    max_iterations=sys.maxsize,
    report_all=False,
    analysis_kind=(
        AnalysisKind.PEP316,
        # AnalysisKind.icontract,
        AnalysisKind.asserts,
    ),
)
