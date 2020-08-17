import datetime

import torchexpo_sphinx_theme
from torchexpo import version
from packaging.version import parse
from recommonmark.transform import AutoStructify


parsed_version = parse(version.__version__)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.programoutput',
    'recommonmark',
]

templates_path = ['_templates']

source_suffix = ['.rst', '.md']

master_doc = 'index'

project = 'TorchExpo'
copyright = str(datetime.datetime.now().year) + ', Omkar Prabhu'
author = 'Omkar Prabhu'

version = parsed_version.base_version
release = str(parsed_version)

language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

todo_include_todos = True

html_theme = 'torchexpo_sphinx_theme'
html_theme_path = [torchexpo_sphinx_theme.get_html_theme_path()]
templates_path = ['_templates']

html_theme_options = {
    'includehidden': False,
    'canonical_url': 'https://torchexpo.readthedocs.io/',
    'pytorch_project': 'docs',
}

html_static_path = ['_static']

html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
    ]
}

html_baseurl = '/'

htmlhelp_basename = 'torchexpodoc'

github_doc_root = 'https://github.com/torchexpo/torchexpo/tree/master'

def setup(app):
    app.add_config_value(
        'recommonmark_config',
        {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
        },
        True,
    )
    app.add_transform(AutoStructify)