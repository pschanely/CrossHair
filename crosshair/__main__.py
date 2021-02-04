"""Run CrossHair as Python module."""

import crosshair.main

if __name__ == "__main__":
    # The ``prog`` needs to be set in the argparse.
    # Otherwise the program name in the help shown to the user will be ``__main__``.
    crosshair.main.main()
