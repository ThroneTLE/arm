#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="competition_sim_bridge",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
