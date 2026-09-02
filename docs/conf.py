"""Sphinx configuration for the public CCW API documentation."""

from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "Clone-Censor-Weight"
author = "Yuki Takazawa and Yuya Kimura"

try:
    release = distribution_version("clone-censor-weight")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
