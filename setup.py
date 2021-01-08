from setuptools import setup, find_packages
import unittest

setup(
    name='crosshair-tool',
    version='0.0.8',
    author='Phillip Schanely',
    author_email='pschanely+vE7F@gmail.com',
    packages=find_packages(),
    scripts=[],
    entry_points = {
        'console_scripts': ['crosshair=crosshair.main:main'],
    },
    url='https://github.com/pschanely/CrossHair',
    license='MIT',
    description='A static analysis tool for Python using symbolic execution.',
    long_description=open('README.md').read().replace('doc/', 'https://raw.githubusercontent.com/pschanely/CrossHair/master/doc/'),
    long_description_content_type='text/markdown',
    install_requires=[
        'forbiddenfruit',
        'typing-inspect',
        'z3-solver==4.8.9.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Testing',
    ],
    python_requires='>=3.7',
)
