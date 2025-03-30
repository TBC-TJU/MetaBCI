# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MetaBCI'
copyright = '2023, TBC-TJU'
author = 'TBC-TJU'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinxcontrib.apidoc',
    'sphinx.ext.viewcode',
    'recommonmark'
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ['.rst', '.md']

import os
import sys
project_path = '../../metabci'
sys.path.insert(0, os.path.abspath(project_path))

apidoc_module_dir = project_path
apidoc_output_dir = 'python_apis'
apidoc_excluded_paths = []
apidoc_separate_modules = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
master_doc = 'index'