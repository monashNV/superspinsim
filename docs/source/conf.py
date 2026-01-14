import os
import sys

project = 'superspinsim'
copyright = '2025, Alex Tritt'
author = 'Alex Tritt'
release = '1.0.1'

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc"
]

templates_path = ["_templates"]
exclude_patterns = []

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../.."))

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"

html_theme_options = {
    "style_nav_header_background": "grey",
}
