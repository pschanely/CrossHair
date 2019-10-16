from setuptools import setup, find_packages

# TODO: using a cusom build of z3 right now - eseentially
# no official release passes all the tests.
setup(
    name='CrossHair',
    version='0.0.1',
    author='Phillip Schanely',
    author_email='pschanely+vE7F@gmail.com',
    packages=['crosshair'],
    scripts=[],
    entry_points = {
        'console_scripts': ['crosshair=crosshair.main:main'],
    },
    url='https://github.com/pschanely/CrossHair',
    license='MIT',
    description='A static analysis tool for Python using symbolic execution.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'typeguard',
        'typing-inspect',
        'z3-solver==4.8.6.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.7',
)
