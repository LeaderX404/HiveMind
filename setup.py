#!/usr/bin/env python3
"""
HiveMind - Neural Swarm Inference for LLMs on low-end devices
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hivemind",
    version="0.1.0",
    description="Run 7B+ LLMs on low-end devices via Neural Swarm Inference (2-bit quantization + distributed computing)",
    author="LeaderX404",
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
)
