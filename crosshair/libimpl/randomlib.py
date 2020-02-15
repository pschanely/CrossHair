import random
from crosshair import register_patch, register_type
from crosshair import realize, with_realized_args, IgnoreAttempt

def make_registrations() -> None:
    register_type(random.Random, lambda p: random.Random(p(int)))

    # # TODO: consider realizing seeds provided to random. Strangely, it seems
    # # like we need to patch both __init__ and __new__.
    #register_patch(random.Random.__init__, with_realized_args(random.Random.__init__), '__init__')
    #register_patch(random._random.Random.__new__, with_realized_args(random._random.Random.__new__), '__new__')  # type: ignore
