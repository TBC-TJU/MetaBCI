# -*- coding: utf-8 -*-
"""
Backward compatibility shim for older pip versions.
All configuration is now in pyproject.toml.

For modern pip (>=21.3), you can install directly with:
    pip install .
    pip install .[brainda]
    pip install .[all]
"""
import setuptools

if __name__ == "__main__":
    setuptools.setup()
