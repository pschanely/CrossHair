from urllib.parse import urlparse

from crosshair.test_util import check_fail


def test_urllib_parse():
    def f(urlstring: str):
        """
        pre: len(urlstring) == 3
        pre: urlstring[0:2] == "//"
        pre: urlstring[2] != "["
        pre: urlstring[2] != "]"
        post: __return__.netloc != "h"
        """
        return urlparse(urlstring)

    (expected, actual) = check_fail(f)
    assert expected == actual
