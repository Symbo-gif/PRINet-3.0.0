# -- PRINet Sphinx Configuration -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project source to path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import prinet  # noqa: E402

# -- Project information ---------------------------------------------------
project = "PRINet"
author = "Michael Maillet, Damien Davison, Sacha Davison"
copyright = "2026, Michael Maillet, Damien Davison, Sacha Davison"
version = prinet.__version__
release = prinet.__version__

# -- General configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc settings -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# Napoleon (Google/NumPy docstring parsing)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST (Markdown support)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Intersphinx -----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML output -----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
}
html_static_path = ["_static"]
html_title = f"PRINet {version}"
html_short_title = "PRINet"
html_logo = None
html_favicon = None
