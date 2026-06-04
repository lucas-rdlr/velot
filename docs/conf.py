# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root to sys.path so Sphinx can find the velot package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "VelOT"
copyright = "2026, Lucas Rincon de la Rosa"
author = "Lucas Rincon de la Rosa"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",           # Pull docstrings from code
    "sphinx.ext.autosummary",       # Generate summary tables
    "sphinx.ext.napoleon",          # Support NumPy/Google docstrings
    "sphinx.ext.viewcode",          # Add [source] links
    "sphinx.ext.intersphinx",       # Link to other projects' docs
    "sphinx.ext.mathjax",           # Render math in docs
    "sphinx_autodoc_typehints",     # Better type hint rendering
    "myst_parser",                  # Support Markdown files
    "nbsphinx",
    'sphinx_copybutton'
]

# -- nbsphinx configuration ---------------------------------------------------
nbsphinx_execute = "never"  # Notebooks are pre-executed; don't re-run on build

# Prolog: added to the top of every notebook rendered in docs
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div style="margin-bottom: 20px; padding: 10px; background-color: #f0f0f0;
                border-radius: 5px; border-left: 4px solid #2980b9;">
        <strong>📓 Interactive notebook</strong><br>
        <a href="https://github.com/lucas-rdlr/velot/tree/main/docs/{{ docname }}"
           target="_blank" style="margin-right: 15px;">
            🔗 View on GitHub
        </a>
        <a href="{{ docname.split('/')[-1] }}.ipynb"
           download style="margin-right: 15px;">
            ⬇️ Download notebook
        </a>
    </div>
"""

# Enable math in MyST markdown
myst_enable_extensions = [
    "dollarmath",      # allows $inline$ and $$block$$ math
    "amsmath",         # allows \begin{equation} etc.
]

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# Intersphinx: link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": True,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#2980b9",
}

html_title = "VelOT Documentation"
html_short_title = "VelOT"

# If you have a logo, uncomment:
# html_logo = "_static/logo.png"

html_static_path = ["_static"]

# Suppress warnings for missing references to torch types
nitpick_ignore = [
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.nn.Module"),
    ("py:class", "torch.device"),
]

# Mock imports for packages that may not be installed in the docs environment
autodoc_mock_imports = [
    "torch",
    "scvelo",
    "ot",
    "ipywidgets",
]