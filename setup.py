# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os.path import join

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="valiml",
    version="0.1.16",
    description="Extension of sklearn with tweaked implementation of common machine learning algorithms for self-use.",
    license="MIT",
    author="Valentin-Bogdan Rosca",
    packages=find_packages(),
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
    setup_requires=requirements,
    install_requires=["cffi"],
    cffi_modules=[join('valiml', 'utils', 'src', 'cffi.py:ffibuilder')],
    zip_safe=True
)
