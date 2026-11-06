#!/usr/bin/env python3
"""
HiveMind by VALCRYPTIA - Out-of-Core Inference Engine for LLMs
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hivemind-valcryptia",
    version="2.1.0",
    description="Out-of-Core Inference Engine for Running 7B+ LLMs on Single Devices with Limited RAM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="VALCRYPTIA",
    author_email="",
    url="https://github.com/LeaderX404/HiveMind",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hivemind=hivemind.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Python 3.8 is EOL and unsupported by torch>=2.6.0, so the
    # floor moves to 3.9 together with the torch security floor.
    python_requires=">=3.9",
)
