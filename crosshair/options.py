import collections
import enum
import math
import re
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, get_type_hints


class AnalysisKind(enum.Enum):
    asserts = "asserts"
    PEP316 = "PEP316"
    icontract = "icontract"
    deal = "deal"
    hypothesis = "hypothesis"

    def __repr__(self):
        return f"AnalysisKind.{self.name}"


def _parse_analysis_kind(argstr: str) -> Sequence[AnalysisKind]:
    try:
        return [AnalysisKind[part.strip()] for part in argstr.split(",")]
    except KeyError:
        raise ValueError


def _parse_bool(argstr: str) -> Optional[bool]:
    match = re.fullmatch(r"(1|true|y(?:es)?)|(0|false|no?)", argstr, re.I)
    if match:
        yes, _no = match.groups()
        return bool(yes)
    return None


@dataclass
class AnalysisOptionSet:
    """
    Encodes some set of partially-specified options.

    This class is used while parsing options from various places.
    It is very similar to `AnalysisOptions` (which is used during execution) but allows
    None values everywhere so that options can correctly override each other.
    """

    analysis_kind: Optional[Sequence[AnalysisKind]] = None
    enabled: Optional[bool] = None
    specs_complete: Optional[bool] = None
    per_condition_timeout: Optional[float] = None
    per_path_timeout: Optional[float] = None
    max_iterations: Optional[int] = None
    report_all: Optional[bool] = None
    report_verbose: Optional[bool] = None
    timeout: Optional[float] = None
    max_uninteresting_iterations: Optional[int] = None

    # TODO: move stats out of options
    stats: Optional[collections.Counter] = None

    # These options are the ones allowed in directives
    directive_fields = frozenset(
        {
            "enabled",
            "analysis_kind",
            "specs_complete",
            "max_iterations",
            "per_condition_timeout",
            "per_path_timeout",
            "max_uninteresting_iterations",
        }
    )

    def overlay(self, overrides: "AnalysisOptionSet") -> "AnalysisOptionSet":
        kw = {k: v for (k, v) in overrides.__dict__.items() if v is not None}
        return replace(self, **kw)

    @classmethod
    def parser_for(cls, field: str) -> Optional[Callable[[str], Any]]:
        if field == "analysis_kind":
            return _parse_analysis_kind
        hints = get_type_hints(AnalysisOptions)
        if field not in hints:
            return None
        ctor = hints[field]
        if ctor is bool:
            return _parse_bool
        return ctor

    @classmethod
    def parse_field(cls, field: str, strval: str) -> Any:
        parser = cls.parser_for(field)
        if parser is None:
            return None
        try:
            return parser(strval)
        except ValueError:
            return None


def option_set_from_dict(source: Mapping[str, object]) -> AnalysisOptionSet:
    options = AnalysisOptionSet()
    for optname in (
        "analysis_kind",
        "specs_complete",
        "per_path_timeout",
        "per_condition_timeout",
        "max_uninteresting_iterations",
        "report_all",
        "report_verbose",
    ):
        arg_val = source.get(optname, None)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options


@dataclass
class AnalysisOptions:
    """Encodes the options for use while running CrossHair."""

    analysis_kind: Sequence[AnalysisKind]
    enabled: bool
    specs_complete: bool
    per_condition_timeout: float
    max_iterations: int
    report_all: bool
    report_verbose: bool
    timeout: float
    per_path_timeout: float
    max_uninteresting_iterations: int

    # Transient members (not user-configurable):
    deadline: float = float("NaN")
    stats: Optional[collections.Counter] = None

    def get_max_uninteresting_iterations(self):
        max_uninteresting_iterations = self.max_uninteresting_iterations
        if max_uninteresting_iterations == sys.maxsize and (
            not math.isfinite(self.per_condition_timeout)
        ):
            return 5
        elif max_uninteresting_iterations == 0:
            return sys.maxsize
        else:
            return max_uninteresting_iterations

    def get_per_path_timeout(self):
        if math.isnan(self.per_path_timeout):
            if math.isfinite(self.per_condition_timeout):
                if self.per_condition_timeout > 1.0:
                    return self.per_condition_timeout**0.5
                else:
                    return self.per_condition_timeout
            max_uninteresting_iterations = self.get_max_uninteresting_iterations()
            if max_uninteresting_iterations < sys.maxsize:
                return max(max_uninteresting_iterations, 1)
            return float("inf")
        else:
            return self.per_path_timeout

    def overlay(
        self, overrides: Optional[AnalysisOptionSet] = None, **kw
    ) -> "AnalysisOptions":
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
    analysis_kind=(
        AnalysisKind.PEP316,
        AnalysisKind.icontract,
        AnalysisKind.deal,
    ),
    enabled=True,
    specs_complete=False,
    per_condition_timeout=float("inf"),
    max_iterations=sys.maxsize,
    report_all=False,
    report_verbose=True,
    timeout=float("inf"),
    per_path_timeout=float("NaN"),
    max_uninteresting_iterations=sys.maxsize,
)
