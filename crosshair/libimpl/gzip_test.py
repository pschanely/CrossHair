import gzip
import json
from io import BytesIO

from crosshair.tracers import ResumedTracing


def test_gzip(space):
    buf = BytesIO()
    with ResumedTracing():
        with gzip.GzipFile(mode="wb", fileobj=buf) as fh:
            fh.write(json.dumps([]).encode("ascii"))
        buf.seek(0)
        with gzip.GzipFile(mode="rb", fileobj=buf) as fh:
            assert json.load(fh) == []
