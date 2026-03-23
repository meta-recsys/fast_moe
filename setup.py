# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
FastMoE Setup Script

This setup.py is provided for backward compatibility.
Modern installations should use pyproject.toml with pip >= 21.3:

    pip install .

For development:

    pip install -e .
"""

from setuptools import setup

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "High-performance PyTorch library for Mixture of Experts"

# Main setup configuration is in pyproject.toml
# This file exists for backward compatibility only
setup(
    name="fast-moe",
    use_scm_version=False,  # Version is in pyproject.toml
    long_description=long_description,
    long_description_content_type="text/markdown",
)
