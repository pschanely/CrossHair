import json

import pytest

from crosshair.core_and_libs import standalone_statespace


def test_disallow_unicode_digits():
    with standalone_statespace:
        float("0E٠")  # This is a valid float!
        with pytest.raises(json.JSONDecodeError):
            json.loads("0E٠")  # But not a valid JSON float.
