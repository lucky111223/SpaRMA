import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SpaRMA"
copyright = "2026, SpaRMA contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "nbsphinx", "nbsphinx_link"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
nbsphinx_execute = "never"
autodoc_mock_imports = ["numpy", "scipy", "sklearn", "torch", "torch_geometric"]
