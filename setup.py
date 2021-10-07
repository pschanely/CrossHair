from setuptools import setup, find_packages  # type: ignore

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
    version="0.0.17",  # Update this in crosshair/__init__.py too
    author="Phillip Schanely",
    author_email="pschanely+vE7F@gmail.com",
    packages=find_packages(),
    scripts=[],
    entry_points={
        "console_scripts": [
            "crosshair=crosshair.main:main",
            "mypycrosshair=crosshair.main:mypy_and_check",
        ],
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
        "typing-inspect>=0.7.1",
        "typing_extensions>=3.10.0",
        "z3-solver==4.8.9.0",
        "importlib_metadata>=4.0.0",
    ],
    extras_require={
        "dev": [
            "autodocsumm>=0.2.2,<1",
            "black==20.8b1",
            "deal>=4.13.0",
            "flake8",
            "hypothesis>=6.0.0",
            "icontract>=2.4.0",
            "mypy==0.902",
            "numpy",  # For doctests in example code
            "pydantic",  # For unittesting (pure vs compiled) Cython imports
            "pydocstyle==5.1.1",
            "pytest",
            "sphinx>=3.4.3,<4",
            "sphinx-autodoc-typehints>=1.11.1",
            "sphinx-rtd-theme>=0.5.1,<1",
            "types-pkg_resources",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.7",
)
