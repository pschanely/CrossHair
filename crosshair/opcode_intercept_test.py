from crosshair.statespace import MessageType
from crosshair.test_util import check_states
from crosshair.core import proxy_for_type


def test_dict_index():
    a = {"two": 2, "four": 4, "six": 6}

    def numstr(x: str) -> int:
        """
        post: _ != 4
        raises: KeyError
        """
        return a[x]

    assert check_states(numstr) == {MessageType.POST_FAIL}
