from crosshair.util import set_debug
from sys import argv

def pytest_configure(config):
    if '-v' in argv or '-vv' in argv:
        set_debug(True)
