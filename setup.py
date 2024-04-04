#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="radium",
    version="2.0.0",
    description="Automated, robust co-design",
    author="Charles Dawson",
    author_email="cbd@mit.edu",
    url="https://github.com/MIT-REALM/radium_tro_24",
    install_requires=[],
    package_data={"radium": ["py.typed"]},
    packages=find_packages(),
)
