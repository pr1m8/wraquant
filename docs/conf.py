# Configuration file for the Sphinx documentation builder.
#
# wraquant — The ultimate quant finance toolkit for Python
# https://wraquant.readthedocs.io

from __future__ import annotations

import importlib.metadata
import os
import sys

# Add the source tree so autodoc can import wraquant
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Version (single source from pyproject.toml) ----------------------------

try:
    release = importlib.metadata.version("wraquant")
except importlib.metadata.PackageNotFoundError:
    release = "1.0.0"

version = ".".join(release.split(".")[:2])

# -- Project information -----------------------------------------------------

project = "wraquant"
copyright = "2024-2026, Algebraic Wealth"
author = "William Astley"

# -- Mock optional dependencies for RTD builds --------------------------------

autodoc_mock_imports = [
    # Volatility / regimes
    "arch",
    "hmmlearn",
    "pomegranate",
    "filterpy",
    "pykalman",
    "dynamax",
    "river",
    "ruptures",
    "stumpy",
    "tsfresh",
    "sktime",
    "tslearn",
    "pywt",
    # Optimization
    "cvxpy",
    "cvxopt",
    "osqp",
    "scs",
    "clarabel",
    "qpsolvers",
    "pulp",
    "pymoo",
    "pyomo",
    "ortools",
    # Bayesian
    "pymc",
    "arviz",
    "bambi",
    "numpyro",
    "blackjax",
    "emcee",
    "dynesty",
    "pyro",
    "cmdstanpy",
    "pytensor",
    # Deep learning
    "torch",
    "torchsde",
    # Pricing
    "QuantLib",
    "financepy",
    "rateslib",
    "mibian",
    "py_vollib",
    # Data
    "yfinance",
    "fredapi",
    "nasdaqdatalink",
    "exchange_calendars",
    "pandas_market_calendars",
    # Backtesting
    "vectorbt",
    "quantstats",
    "empyrical",
    "pyfolio",
    "alphalens",
    "ffn",
    # Risk
    "pypfopt",
    "riskfolio",
    "skfolio",
    "copulas",
    "copulae",
    "pyvinecopulib",
    "pyextremes",
    # Visualization
    "plotly",
    "bokeh",
    "altair",
    "seaborn",
    "matplotlib",
    "kaleido",
    # Workflow / scale
    "streamlit",
    "prefect",
    "dagster",
    "apscheduler",
    "dask",
    "ray",
    # Causal
    "dowhy",
    "econml",
    "doubleml",
    # Cleaning / validation
    "pandera",
    "pyjanitor",
    "rapidfuzz",
    "dateparser",
    # Math / JAX
    "jax",
    "jaxlib",
    "jaxopt",
    "optax",
    "equinox",
    "diffrax",
    "sympy",
    "symengine",
    # Misc
    "polars",
    "duckdb",
    "numba",
    "sqlalchemy",
    "connectorx",
    "httpx",
    "aiohttp",
    "websockets",
    "sdeint",
    "sdepy",
    "darts",
    "shap",
    "loguru",
    # MCP
    "fastmcp",
    "joblib",
]

# -- General configuration ---------------------------------------------------

extensions = [
    # Core Sphinx
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # Third-party
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_class_signature = "separated"

# -- Type hints settings -----------------------------------------------------

typehints_fully_qualified = False
always_document_param_types = True
typehints_defaults = "comma"

# -- Autosummary settings ----------------------------------------------------

autosummary_generate = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

# -- MathJax settings --------------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "E": r"\mathbb{E}",
            "Var": r"\mathrm{Var}",
            "Cov": r"\mathrm{Cov}",
            "VaR": r"\mathrm{VaR}",
            "CVaR": r"\mathrm{CVaR}",
        }
    }
}

# -- MyST-Parser settings ----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# -- Copy button settings ----------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- TODO settings -----------------------------------------------------------

todo_include_todos = False

# -- Theme configuration -----------------------------------------------------

html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448AFF",
        "color-brand-content": "#448AFF",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/pr1m8/wraquant",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = f"wraquant v{release}"
html_short_title = "wraquant"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Source suffix -----------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "_internal", "Thumbs.db", ".DS_Store"]

# -- Misc settings -----------------------------------------------------------

add_module_names = False
nitpicky = False
default_role = "py:obj"
pygments_style = "monokai"
pygments_dark_style = "monokai"
