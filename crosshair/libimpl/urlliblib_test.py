from urllib.parse import urlparse

from crosshair.options import AnalysisOptionSet
from crosshair.statespace import POST_FAIL, MessageType
from crosshair.test_util import check_states


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

    check_states(f, POST_FAIL)
