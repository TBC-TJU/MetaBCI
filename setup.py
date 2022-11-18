# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", errors="ignore", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="MetaBCI",
    version="0.1.0",
    author="TBC-TJU",
    author_email="TBC_TJU_2022@163.com",
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