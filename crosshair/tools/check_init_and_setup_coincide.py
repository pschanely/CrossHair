#!/usr/bin/env python3

"""Check that the distribution and crosshair/__init__.py are in sync."""
import subprocess
import sys
from typing import Dict, Optional

import crosshair


def main() -> int:
    """Execute the main routine."""
    success = True

    ##
    # Check basic fields
    ##

    setup_py = dict()  # type: Dict[str, str]

    fields = ["version", "author", "license", "description"]
    for field in fields:
        out = subprocess.check_output(
            [sys.executable, "setup.py", f"--{field}"], encoding="utf-8"
        ).strip()

        setup_py[field] = out

    if setup_py["version"] != crosshair.__version__:
        print(
            f"The version in the setup.py is {setup_py['version']}, "
            f"while the version in crosshair/__init__.py is: "
            f"{crosshair.__version__}",
            file=sys.stderr,
        )
        success = False

    if setup_py["author"] != crosshair.__author__:
        print(
            f"The author in the setup.py is {setup_py['author']}, "
            f"while the author in crosshair/__init__.py is: "
            f"{crosshair.__author__}",
            file=sys.stderr,
        )
        success = False

    if setup_py["license"] != crosshair.__license__:
        print(
            f"The license in the setup.py is {setup_py['license']}, "
            f"while the license in crosshair/__init__.py is: "
            f"{crosshair.__license__}",
            file=sys.stderr,
        )
        success = False

    if setup_py["description"] != crosshair.__doc__:
        print(
            f"The description in the setup.py is {setup_py['description']}, "
            f"while the description in crosshair/__init__.py is: "
            f"{crosshair.__doc__}",
            file=sys.stderr,
        )
        success = False

    ##
    # Classifiers need special attention as there are multiple.
    ##

    # This is the map from the distribution to expected status in __init__.py.
    status_map = {
        "Development Status :: 1 - Planning": "Planning",
        "Development Status :: 2 - Pre-Alpha": "Pre-Alpha",
        "Development Status :: 3 - Alpha": "Alpha",
        "Development Status :: 4 - Beta": "Beta",
        "Development Status :: 5 - Production/Stable": "Production/Stable",
        "Development Status :: 6 - Mature": "Mature",
        "Development Status :: 7 - Inactive": "Inactive",
    }

    classifiers = (
        subprocess.check_output(
            [sys.executable, "setup.py", f"--classifiers"], encoding="utf-8"
        )
        .strip()
        .splitlines()
    )

    status_classifier = None  # type: Optional[str]
    for classifier in classifiers:
        if classifier in status_map:
            status_classifier = classifier
            break

    if status_classifier is None:
        print(
            f"Expected a status classifier in setup.py "
            f"(e.g., 'Development Status :: 3 - Alpha'), but found none.",
            file=sys.stderr,
        )
        success = False
    else:
        expected_status_in_init = status_map[status_classifier]

        if expected_status_in_init != crosshair.__status__:
            print(
                f"Expected status {expected_status_in_init} "
                f"according to setup.py in crosshair/__init__.py, "
                f"but found: {crosshair.__status__}"
            )
            success = False

    if not success:
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())
