# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IRTorch'
copyright = '2024, Joakim Wallmark'
author = 'Joakim Wallmark'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx', # Support for Jupyter Notebooks
    'sphinx.ext.duration', # Support for durations in the format 1h 30m 15s when using building the documentation
    'sphinx.ext.autodoc', # Automatically document code
    'sphinx.ext.napoleon', # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.mathjax', # Support for LaTeX math
    'sphinxcontrib.bibtex' # Support for BibTeX
]

# nbsphinx_execute = 'always'

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
