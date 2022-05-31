from importlib._bootstrap import _find_and_load_unlocked  # type: ignore

from crosshair import register_patch
from crosshair.tracers import NoTracing


# Do not import with tracing enabled.
# (it's expensive, and won't behave deterministically)
#
# NOTE: The importlib._bootstrap._find_and_load_unlocked entry point covers both the
# regular import statement, as well as loads via the importlib module.
def __find_and_load_unlocked(*a, **kw):
    with NoTracing():
        return _find_and_load_unlocked(*a, **kw)


def make_registrations() -> None:
    register_patch(_find_and_load_unlocked, __find_and_load_unlocked)
