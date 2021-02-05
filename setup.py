from setuptools import setup, find_packages

# Do not forget to update and sync the fields in crosshair/__init__.py!
#
# (mristin, 2021-02-05): It is almost impossible to refer to the
# crosshair/__init__.py from within setup.py as the source distribution will
# run setup.py while installing the package.
# That is why we can not be DRY here and need to sync manually.
#
# See also this StackOverflow question:
# https://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#
# The fields between crosshair/__init__.py and this file are checked as part of
# the pre-commit checks through check_init_and_setup_coincide.py.
setup(
    name="crosshair-tool",
    version="0.0.9",
    author="Phillip Schanely",
    author_email="pschanely+vE7F@gmail.com",
    packages=find_packages(),
    scripts=[],
    entry_points={
        "console_scripts": ["crosshair=crosshair.main:main"],
    },
    url="https://github.com/pschanely/CrossHair",
    license="MIT",
    description="Analyze Python code for correctness using symbolic execution.",
    long_description=open("README.md")
    .read()
    .replace(
        "doc/", "https://raw.githubusercontent.com/pschanely/CrossHair/master/doc/"
    ),
    long_description_content_type="text/markdown",
    install_requires=[
        "forbiddenfruit",
        "typing-inspect",
        "z3-solver==4.8.9.0",
    ],
    extras_require={
        "dev": [
            "icontract",
            "numpy",  # For doctests in example code
            "pytest",
            "flake8",
            "coverage",
            "codecov",
            "black==20.8b1",
            "pydocstyle==5.1.1",
            "wheel",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.7",
)
