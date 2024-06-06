# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", errors="ignore", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="metabci",
    version="0.1.2",
    author="TBC-TJU",
    author_email="TBC_TJU_2022@163.com",
    description="A Library of Datasets, Algorithms, \
        and Experiments workflow for Brain-Computer Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        'setuptools',
        'wheel',
        'twine',
        'flake8',
        'mypy',
        'coverage',
        'mat73',
        'tqdm>=4.32.0',
        'torch>=1.7.1',
        'numpy',
        'mne>=0.21.1',
        'pandas',
        'py7zr',
        'joblib',
        'autograd',
        'scipy',
        'pymanopt==0.2.5',
        'requests',
        'requests[socks]',
        'pytest',
        'h5py',
        'scikit_learn',
        'sphinxcontrib-napoleon',
        'skorch',
        'pooch',
        'pylsl',
        'wxPython==4.1.1; \
        sys_platform == \'darwin\' and python_version <= \'3.8\'',
        'pyglet==1.5.27; \
        sys_platform == \'darwin\' and python_version <= \'3.8\'',
        'psychopy'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
