# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", errors="ignore", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="metabci",
    version="0.1",
    author="TUNERL",
    author_email="swolfforever@gmail.com",
    description="A Library of Datasets, Algorithms, and Experiments workflow for Brain-Computer Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)