from typing import Tuple, Type

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from crosshair import IgnoreAttempt, SymbolicFactory, realize, register_type

#
# Classes implemented in C generally cannot be simulated symbolically by
# CrossHair.
# However, you can install hooks to produce custom symbolic values.
# Here, we provide a lazy version of the numpy's array class: this has
# a symbolic shape and data type.
# When an actual operation needs to be performed, we'll construct the
# actual array.
#


class SymbolicNdarray(NDArrayOperatorsMixin):
    def __init__(self, creator: SymbolicFactory):
        # Our callback gets a SymbolicFactory instance which can produce more
        # symbolic values when called with a type.
        self.shape = creator(Tuple[int, ...], "_shape")
        # Note that we avoid the builtin len() - symbolic creation hooks do not run
        # under the monkeypatched environment, and calling the real len() would
        # realize the shape's length.
        self.ndim = self.shape.__len__()
        self.dtype = np.dtype(realize(creator(Type[np.number], "_dtype")))

    @property
    def size(self):
        totalsize = 1
        for size in self.shape:
            if size < 0:
                raise IgnoreAttempt("ndarray disallows negative dimensions")
            totalsize *= size
        return totalsize

    def __repr__(self):
        return repr(self.__array__())

    def _realize_args(self, args):
        newargs = []
        for arg in args:
            if isinstance(arg, self.__class__):
                newargs.append(arg.__array__())
            else:
                # Call realize() for operations on symbolic floats, etc:
                newargs.append(realize(arg))
        return newargs

    def __array_function__(self, func, types, args, kwargs):
        return func(*self._realize_args(args), **kwargs)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == "__call__":
            return ufunc(*self._realize_args(args), **kwargs)
        else:
            return NotImplemented

    def __array__(self):
        if any(size < 0 for size in self.shape):
            raise IgnoreAttempt("ndarray disallows negative dimensions")
        concrete_shape = tuple(map(int, self.shape))
        concrete_dtype = realize(self.dtype)
        # For the contents, we just construct it with ones. This makes it much
        # less complete in terms of finding counterexamples, but is sufficient
        # for array dimension and type reasoning. If we were more ambitious,
        # we would rewrite a (slow) implementation of numpy in terms of native
        # Python types.
        return np.ones(concrete_shape, concrete_dtype)


# Make crosshair use our custom class whenever it needs a symbolic
# ndarray instance:
register_type(np.ndarray, SymbolicNdarray)


def matrix_multiply(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    pre: image1.dtype == image2.dtype == np.float64
    pre: len(image1.shape) == len(image2.shape) == 2
    pre: image1.shape[1] == image2.shape[0]
    post: __return__.shape == (image1.shape[0], image2.shape[1])
    """
    return image1 @ image2


def unit_normalize(a: np.ndarray) -> np.ndarray:
    """
    Normalize the given array values into the [0,1] range.

    >>> unit_normalize(np.arange(-1, 2))
    array([0. , 0.5, 1. ])

    pre: a.size > 0
    pre: a.dtype == np.float64
    pre: np.ptp(a) > 0
    post: np.max(_) <= 1.0
    post: np.min(_) >= 0.0
    """
    return (a - np.min(a)) / np.ptp(a)


def threshold_image(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    >>> threshold_image(np.array([[0.0, 0.3], [0.6, 1.0]], dtype=np.float64), 0.5)
    array([[0.5, 0.5],
           [0.6, 1. ]])

    pre: len(image.shape) == 2
    pre: image.dtype == np.float64
    pre: image.size > 0
    pre: threshold > 0
    post: _.shape == image.shape
    post: image.dtype == _.dtype
    post: np.min(_) >= threshold
    """
    return np.where(image > threshold, image, threshold)
