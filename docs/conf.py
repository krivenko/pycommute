# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from distutils.util import get_platform
from distutils.sysconfig import get_python_version

sys.path.insert(0, os.path.abspath(os.path.join(
    "..", "build", "lib.%s-%s" % (get_platform(), get_python_version())))
)

sys.path.append(os.path.abspath("./_ext"))

# -- Project information -----------------------------------------------------

project = 'pycommute'
copyright = '2020-2021, Igor Krivenko'
author = 'Igor Krivenko'
version = '0.7.1'
release = '0.7.1'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '2.1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'myst_parser',
    'pybind11_preprocess'
]

# The suffix of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "images/logo_small.svg"
