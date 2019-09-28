import collections.abc
from typing import MutableSequence

_MISSING = object()

class SimpleDict(collections.abc.MutableMapping):
    '''
    #inv: set(self.keys()) == set(dict(self.items()).keys())

    >>> d = SimpleDict([(1, 'one'), (2, 'two')])
    >>> d
    {1: 'one', 2: 'two'}
    >>> d[3] = 'three'
    >>> len(d)
    3
    >>> d[2] = 'cat'
    >>> d[2]
    'cat'
    >>> del d[1]
    >>> list(d.keys())
    [2, 3]
    '''
    contents_: MutableSequence
    def __init__(self, contents: MutableSequence):
        # TODO: assumes initial data has no duplicate keys. Is that right?
        self.contents_ = contents
    def __getitem__(self, key, default=_MISSING):
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                return v
        if default is _MISSING:
            raise KeyError(key)
        return default
    def __setitem__(self, key, value):
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                self.contents_[i] = (k, value)
                return
        self.contents_.append((key, value))
    def __delitem__(self, key):
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                del self.contents_[i]
                return
    def  __iter__(self):
        return (k for (k, v) in self.contents_)
    def  __bool__(self):
        return (len(self.contents_) > 0).__bool__()
    def  __len__(self):
        '''
        post: _ >= 0
        '''
        return len(self.contents_)
    def __repr__(self):
        return str(dict(self.items()))

