import sys
import urllib.parse


class DevNullDict:
    def get(self, k, default):
        return default

    def __setitem__(self, k, v):
        pass

    def __len__(self):
        return 0


def make_registrations():
    # Avoid nondeterminism in urllib by disabling internal caching.
    if sys.version_info < (3, 11):
        # TODO: This is destructive! But not terribly harmful. Consider alternatives.
        urllib.parse._parse_cache = DevNullDict()  # type: ignore
    else:
        # (3.11 uses functools.lru_cache, which we disable elsewhere)
        assert not hasattr(urllib.parse, "_parse_cache")
