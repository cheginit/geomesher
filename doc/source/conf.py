# Based on xarray's conf.py
# https://github.com/pydata/xarray/blob/main/doc/conf.py

import datetime
import inspect
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from textwrap import dedent, indent

import yaml
from sphinx.application import Sphinx
from sphinx.util import logging

import geomesher

LOGGER = logging.getLogger("conf")

allowed_failures = set()

print("python exec:", sys.executable)
print("sys.path:", sys.path)

if "CONDA_DEFAULT_ENV" in os.environ or "conda" in sys.executable:
    print("conda environment:")
    subprocess.run([os.environ.get("CONDA_EXE", "conda"), "list"])
else:
    print("pip environment:")
    subprocess.run([sys.executable, "-m", "pip", "list"])

print(f"geomesher: {geomesher.__version__}, {geomesher.__file__}")

with suppress(ImportError):
    import matplotlib

    matplotlib.use("Agg")

nbsphinx_allow_errors = False

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx.ext.linkcode",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx_design",
]

extlinks = {
    "issue": ("https://github.com/cheginit/geomesher/issues/%s", "GH%s"),
    "pull": ("https://github.com/cheginit/geomesher/pull/%s", "PR%s"),
}

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# nbsphinx configurations
nbsphinx_timeout = 600
nbsphinx_execute = "always"
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

You can run this notebook in a `live session <https://mybinder.org/v2/gh/cheginit/geomesher/doc/examples/main?urlpath=lab/tree/doc/{{ docname }}>`_ |Binder| or view it `on Github <https://github.com/cheginit/geomesher/blob/main/doc/{{ docname }}>`_.

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/cheginit/geomesher/main?urlpath=lab/tree/doc/{{ docname }}
"""

# autoapi configurations
autoapi_dirs = [
    "../../geomesher",
]
autoapi_ignore = [
    "**ipynb_checkpoints**",
    "**tests**",
    "**conf.py",
    "**conftest.py",
    "**noxfile.py",
    "**exceptions.py",
    "**cli.py",
    "**common.py",
    "**fields.py",
    "**geometry.py",
]
autoapi_options = ["members"]
autoapi_member_order = "groupwise"
# autoapi_keep_files = True
# autoapi_add_toctree_entry = False
modindex_common_prefix = ["geomesher."]

# autodoc_typehints = "none"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Napoleon configurations
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    # special terms
    # "same type as caller": "*same type as caller*",  # does not work, yet
    # "same type as values": "*same type as values*",  # does not work, yet
    # stdlib type aliases
    "MutableMapping": "~collections.abc.MutableMapping",
    "sys.stdout": ":obj:`sys.stdout`",
    "timedelta": "~datetime.timedelta",
    "string": ":class:`string <str>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    "hashable": ":term:`hashable <name>`",
    # matplotlib terms
    "color-like": ":py:func:`color-like <matplotlib.colors.is_color_like>`",
    "matplotlib colormap name": ":doc:`matplotlib colormap name <matplotlib:gallery/color/colormap_reference>`",
    "matplotlib axes object": ":py:class:`matplotlib axes object <matplotlib.axes.Axes>`",
    "colormap": ":py:class:`colormap <matplotlib.colors.Colormap>`",
    # objects without namespace: geomesher
    "DataArray": "~geomesher.DataArray",
    "Dataset": "~geomesher.Dataset",
    "Variable": "~geomesher.Variable",
    "DatasetGroupBy": "~geomesher.core.groupby.DatasetGroupBy",
    "DataArrayGroupBy": "~geomesher.core.groupby.DataArrayGroupBy",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
    "MaskedArray": "~numpy.ma.MaskedArray",
    "dtype": "~numpy.dtype",
    "ComplexWarning": "~numpy.ComplexWarning",
    # objects without namespace: pandas
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "Series": "~pandas.Series",
    "DataFrame": "~pandas.DataFrame",
    "Categorical": "~pandas.Categorical",
    "Path": "~~pathlib.Path",
    # objects with abbreviated namespace (from pandas)
    "pd.Index": "~pandas.Index",
    "pd.NaT": "~pandas.NaT",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
# source_suffix = ".rst"


# The master toctree document.
master_doc = "index"

# General information about the project.
project = "geomesher"
copyright = f"2023-{datetime.datetime.now().year}, Taher Chegini"

# The short X.Y version.
version = geomesher.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = geomesher.__version__

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output ----------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
html_title = ""

html_context = {
    "github_user": "cheginit",
    "github_repo": "geomesher",
    "github_version": "main",
    "doc_path": "doc",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # analytics_id=''  this is configured in rtfd.io
    # canonical_url="",
    "repository_url": "https://github.com/cheginit/geomesher",
    "repository_branch": "main",
    "path_to_docs": "doc",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": False,
    "icon_links": [],  # workaround for pydata/pydata-sphinx-theme#1220
}


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["style.css"]


# configuration for sphinxext.opengraph
ogp_site_url = "https://docs.geomesher.dev/en/latest/"
ogp_image = "https://docs.geomesher.dev/en/stable/_static/logo.png"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image" />',
    '<meta property="twitter:site" content="@xarray_dev" />',
    '<meta name="image" property="og:image" content="https://geomesher.readthedocs.io/en/stable/_static/logo.png" />',
]

# Sometimes the savefig directory doesn't exist and needs to be created
# https://github.com/ipython/ipython/issues/8733
# becomes obsolete when we can pin ipython>=5.2; see ci/requirements/doc.yml
ipython_savefig_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_build", "html", "_static"
)
if not os.path.exists(ipython_savefig_dir):
    os.makedirs(ipython_savefig_dir)


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt

# Output file base name for HTML help builder.
htmlhelp_basename = "geomesherdoc"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandamesh": ("https://deltares.github.io/pandamesh", None),
    "tobler": ("https://pysal.org/tobler", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
}


# based on numpy doc/source/conf.py
def linkcode_resolve(domain: str, info: dict[str, str]):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        source = []
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    fn = os.path.relpath(fn, start=os.path.dirname(geomesher.__file__))

    if "+" in geomesher.__version__:
        return f"https://github.com/cheginit/geomesher/blob/main/geomesher/{fn}{linespec}"
    else:
        return (
            "https://github.com/cheginit/geomesher/blob/"
            f"v{geomesher.__version__}/geomesher/{fn}{linespec}"
        )


def html_page_context(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, bool], doctree: str
) -> None:
    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False


def update_gallery(app: Sphinx):
    """Update the gallery page.

    Taken from https://github.com/pydata/xarray/blob/main/doc/conf.py
    """
    LOGGER.info("Updating gallery page...")

    gallery = yaml.safe_load(Path(app.srcdir, "gallery.yml").read_bytes())
    # for item in gallery:
        # thumb = Path(item["thumbnail"])
        # thumb.parent.mkdir(parents=True, exist_ok=True)
        # thumb.write_bytes(
        #     Path(app.srcdir, "examples", "notebooks", "_static", thumb.name).read_bytes()
        # )

    items = [
        f"""
         .. grid-item-card::
            :text-align: center
            :link: {item['path']}

            .. image:: {item['thumbnail']}
                :alt: {item['title']}
            +++
            {item['title']}
        """
        for item in gallery
    ]

    items_md = indent(dedent("\n".join(items)), prefix="    ")
    markdown = f"""
.. grid:: 1 2 2 2
    :gutter: 2

    {items_md}
"""
    Path(app.srcdir, "examples-gallery.txt").write_text(markdown)
    LOGGER.info("Gallery page updated.")


def setup(app: Sphinx):
    app.connect("html-page-context", html_page_context)
    app.connect("builder-inited", update_gallery)
