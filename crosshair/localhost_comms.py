"""
Provides a way for processes to leave information for other processes
on the same machine. (left as files in the system's tempdir)

>>> with StateUpdater() as updater:
...   updater.update('hello there!')
...   [content for _fname, content in read_states()]
['hello there!']
>>> list(read_states())
[]

"""

import glob
import os.path
import tempfile
from typing import cast, IO, Iterable, Optional, Tuple

from crosshair.util import debug

_PREFIX = "CrossHair_"
_ENCODING = "utf-8"


class StateUpdater:
    """
    inv: self.cur_file is None or not self.cur_file.closed
    """

    cur_file: Optional[IO[str]] = None

    def _close(self):
        """ post[self]: True """
        if self.cur_file is None:
            return
        try:
            self.cur_file.close()
        except:
            debug(f'WARNING: unable to close tmp file "{self.cur_file}"')
        self.cur_file = None

    def update(self, state: str):
        """ post[self]: True """
        new_file = cast(
            IO[str],
            tempfile.NamedTemporaryFile(prefix=_PREFIX, mode="w", encoding=_ENCODING),
        )
        new_file.write(state)
        new_file.flush()
        self._close()
        self.cur_file = new_file

    def __enter__(self) -> "StateUpdater":
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """ post[self]: True """
        self._close()


def read_states() -> Iterable[Tuple[str, str]]:
    path_pattern = os.path.join(tempfile.gettempdir(), _PREFIX + "*")
    for filename in glob.glob(path_pattern):
        if not os.path.exists(filename):
            continue
        try:
            with open(filename, encoding=_ENCODING) as fh:
                yield (filename, fh.read())
        except:
            debug(f'WARNING: unable to read tmp file "{filename}", ignoring.')
            continue
