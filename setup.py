# setup.py
from setuptools import setup, find_packages

setup(
    name="triton-viz",
    version="0.1.0",
    packages=find_packages(exclude=["frontend", "frontend.*"]),
)
