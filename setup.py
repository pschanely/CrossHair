from setuptools import Extension, find_packages, setup  # type: ignore

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
    version="0.0.98",  # Update this in crosshair/__init__.py too
    author="Phillip Schanely",
    author_email="pschanely+vE7F@gmail.com",
    ext_modules=[
        Extension(
            "_crosshair_tracers",
            sources=["crosshair/_tracers.c"],
        ),
    ],
    package_data={"crosshair": ["*.h", "py.typed"]},
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
    long_description=open("README.md", encoding="utf-8")
    .read()
    .replace("doc/", "https://raw.githubusercontent.com/pschanely/CrossHair/main/doc/"),
    long_description_content_type="text/markdown",
    install_requires=[
        "packaging",
        "typing-inspect>=0.7.1",
        "typing_extensions>=3.10.0",
        "z3-solver>=4.13.0.0",
        "importlib_metadata>=4.0.0",
        "pygls>=1.0.0",  # For the LSP server
        "typeshed-client>=2.0.5",
    ],
    extras_require={
        "dev": [
            "autodocsumm>=0.2.2,<1",
            "black==25.9.0",  # sync this with .pre-commit-config.yml
            "deal>=4.13.0",
            "icontract>=2.4.0",
            "isort==5.11.5",  # sync this with .pre-commit-config.yml
            "mypy==1.18.1",  # sync this with .pre-commit-config.yml
            "numpy==1.24.0; python_version < '3.12'",
            "numpy==2.3.3; python_version >= '3.12'",
            "pre-commit~=2.20",
            "pytest",
            "pytest-xdist",
            "setuptools",
            "sphinx>=3.4.3",
            "sphinx-rtd-theme>=0.5.1",
            "rst2pdf>=0.102",
            "z3-solver==4.15.4.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
)
