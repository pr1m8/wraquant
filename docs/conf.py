# Configuration file for the Sphinx documentation builder.
#
# wraquant — The ultimate quant finance toolkit for Python
# https://wraquant.readthedocs.io

from __future__ import annotations

import os
import sys
from datetime import datetime

# Add the source tree so autodoc can import wraquant
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# Mock OPTIONAL dependencies that may not be installed on RTD.
# Core deps (numpy, scipy, pandas, statsmodels, pydantic) are installed
# via the pip install in .readthedocs.yaml.
autodoc_mock_imports = [
    # Volatility / regimes
    "arch", "hmmlearn", "pomegranate", "filterpy", "pykalman", "dynamax",
    "river", "ruptures", "stumpy", "tsfresh", "sktime", "tslearn", "pywt",
    # Optimization
    "cvxpy", "cvxopt", "osqp", "scs", "clarabel", "qpsolvers", "pulp",
    "pymoo", "pyomo", "ortools",
    # Bayesian
    "pymc", "arviz", "bambi", "numpyro", "blackjax", "emcee", "dynesty",
    "pyro", "cmdstanpy", "pytensor",
    # Deep learning
    "torch", "torchsde",
    # Pricing
    "QuantLib", "financepy", "rateslib", "mibian", "py_vollib",
    # Data
    "yfinance", "fredapi", "nasdaqdatalink",
    "exchange_calendars", "pandas_market_calendars",
    # Backtesting
    "vectorbt", "quantstats", "empyrical", "pyfolio", "alphalens", "ffn",
    # Risk
    "pypfopt", "riskfolio", "skfolio", "copulas", "copulae",
    "pyvinecopulib", "pyextremes",
    # Visualization
    "plotly", "bokeh", "altair", "seaborn", "matplotlib", "kaleido",
    # Workflow / scale
    "streamlit", "prefect", "dagster", "apscheduler",
    "dask", "ray",
    # Causal
    "dowhy", "econml", "doubleml",
    # Cleaning / validation
    "pandera", "pyjanitor", "rapidfuzz", "dateparser",
    # Math / JAX
    "jax", "jaxlib", "jaxopt", "optax", "equinox", "diffrax",
    "sympy", "symengine",
    # Misc
    "polars", "duckdb", "numba",
    "sqlalchemy", "connectorx", "httpx", "aiohttp", "websockets",
    "sdeint", "sdepy", "darts", "shap",
    "loguru",
]

# -- Project information -----------------------------------------------------

project = "wraquant"
copyright = f"{datetime.now().year}, Algebraic Wealth"
author = "William Astley"

try:
    from wraquant import __version__

    release = __version__
except ImportError:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

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
    "sphinx.ext.coverage",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    # Third-party
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "myst_parser",
    "sphinx_copybutton",
]

# Try optional extensions (don't fail if not installed)
try:
    import sphinx_design  # noqa: F401

    extensions.append("sphinx_design")
except ImportError:
    pass

try:
    import sphinxcontrib.mermaid  # noqa: F401

    extensions.append("sphinxcontrib.mermaid")
except ImportError:
    pass

# -- AutoAPI settings --------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/wraquant"]
autoapi_ignore = ["*/__pycache__/*", "*/_compat.py"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_keep_files = True
autoapi_add_toctree_entry = True

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True  # Support both styles
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ArrayLike": "wraquant.core.types.ArrayLike",
    "PriceSeries": "wraquant.frame.base.PriceSeries",
    "ReturnSeries": "wraquant.frame.base.ReturnSeries",
    "OHLCVFrame": "wraquant.frame.base.OHLCVFrame",
    "ReturnFrame": "wraquant.frame.base.ReturnFrame",
    "RegimeResult": "wraquant.regimes.base.RegimeResult",
    "GARCHResult": "wraquant.core.results.GARCHResult",
    "BacktestResult": "wraquant.core.results.BacktestResult",
    "ForecastResult": "wraquant.core.results.ForecastResult",
}
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}
autodoc_class_signature = "separated"

# -- Autosummary settings ----------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

# -- Type hints settings -----------------------------------------------------

typehints_fully_qualified = False
always_document_param_types = True
typehints_defaults = "comma"

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "arch": ("https://arch.readthedocs.io/en/latest/", None),
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
            "SR": r"\mathrm{SR}",
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

todo_include_todos = True

# -- Theme configuration -----------------------------------------------------

html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
        "color-admonition-background": "rgba(41, 98, 255, 0.1)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448AFF",
        "color-brand-content": "#448AFF",
        "color-admonition-background": "rgba(68, 138, 255, 0.1)",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/pr1m8/wraquant",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = "wraquant"
html_short_title = "wraquant"
html_favicon = None
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Source suffix -----------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "_internal", "Thumbs.db", ".DS_Store"]

# -- Misc settings -----------------------------------------------------------

# Show typehints in descriptions, not signatures (cleaner)
add_module_names = False

# Suppress warnings for missing references to optional packages
nitpicky = False

# Default role for inline code
default_role = "py:obj"

# Pygments style
pygments_style = "monokai"
pygments_dark_style = "monokai"
