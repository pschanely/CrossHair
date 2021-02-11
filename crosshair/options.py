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
class AnalysisOptions:
    per_condition_timeout: float = 1.5
    per_path_timeout: float = 0.75
    max_iterations: int = sys.maxsize
    report_all: bool = False
    analysis_kind: Sequence[AnalysisKind] = (
        AnalysisKind.PEP316,
        # AnalysisKind.icontract,
        AnalysisKind.asserts,
    )
    timeout: Optional[float] = None

    # Transient members (not user-configurable):
    deadline: float = float("NaN")
    stats: Optional[collections.Counter] = None

    def overlay(self, overrides: "AnalysisOptions") -> "AnalysisOptions":
        return replace(self, **overrides.__dict__)

    @staticmethod
    def from_dict(source: Mapping[str, object]) -> "AnalysisOptions":
        options = AnalysisOptions()
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


DEFAULT_OPTIONS = AnalysisOptions()
