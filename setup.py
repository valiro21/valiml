# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="valiml",
    version="0.1.0",
    description="Extension of sklearn with tweaked implementation of common machine learning algorithms for self-use.",
    license="MIT",
    author="Valentin-Bogdan Roșca",
    packages=find_packages(),
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
    setup_requires=["cffi"],
    install_requires=["cffi"],
    cffi_modules=["valiml/utils/src/cffi.py:ffibuilder"]
)
