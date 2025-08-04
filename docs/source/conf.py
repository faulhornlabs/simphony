# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../simphony'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Simphony'
copyright = '2025, Qutility @ Faulhorn Labs'
author = 'Qutility @ Faulhorn Labs'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
    'myst_nb'
]

autosummary_generate = True

latex_elements = {'preamble': r'\usepackage{mathtools}'}

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'qiskit': ('https://docs.quantum.ibm.com/api/qiskit/', None),
    'qiskit-dynamics': ('https://qiskit-community.github.io/qiskit-dynamics/', None),
}

napoleon_google_docstring = True

# autodoc options
autodoc_member_order = 'bysource'
autodoc_inherit_docstrings = False

#autodoc_default_options = {'special-members': '__call__' }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

add_module_names = False

html_show_sourcelink = False

nb_execution_mode = "off"
nbsphinx_allow_errors = True
myst_enable_extensions = [
    "amsmath",
    "dollarmath"
]

always_use_bars_union = True