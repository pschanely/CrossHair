"""
Configure settings before running pytest

In this case, we set PYTHONHASHSEED to 0 for uniformity across test runs.
"""

import os

os.environ["PYTHONHASHSEED"] = "0"
