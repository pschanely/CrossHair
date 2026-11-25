"""
SMT sorts, terms, solvers, and models.

Names follow SMT-LIB. Constructors never convert Python values into terms;
use the ``*_val`` functions to build literals.

The CROSSHAIR_SMT_BACKEND environment variable selects the solver: "z3" (the
default) or "cvc5".
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING or os.environ.get("CROSSHAIR_SMT_BACKEND", "z3") == "z3":
    from crosshair.smt_z3 import *  # noqa: F401,F403
else:
    from crosshair.smt_cvc5 import *  # noqa: F401,F403
