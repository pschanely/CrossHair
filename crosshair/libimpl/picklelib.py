import pickle

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    register_patch(pickle.dumps, with_realized_args(pickle.dumps, deep=True))
