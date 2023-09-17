.. image:: https://raw.githubusercontent.com/cheginit/geomesher/main/doc/source/_static/logo-text.png
    :target: https://geomesher.readthedocs.io

|

GeoMesher: Meshing a GeoDataFrame using Gmsh
============================================

.. image:: https://github.com/cheginit/geomesher/actions/workflows/test.yml/badge.svg
   :target: https://github.com/cheginit/geomesher/actions/workflows/test.yml
   :alt: CI

.. image:: https://img.shields.io/pypi/v/geomesher.svg
    :target: https://pypi.python.org/pypi/geomesher
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/geomesher.svg
    :target: https://anaconda.org/conda-forge/geomesher
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/geomesher/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/geomesher
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/geomesher.svg
    :target: https://pypi.python.org/pypi/geomesher
    :alt: Python Versions

|

.. image:: https://static.pepy.tech/badge/geomesher
    :target: https://pepy.tech/project/geomesher
    :alt: Downloads

.. image:: https://www.codefactor.io/repository/github/cheginit/geomesher/badge/main
    :target: https://www.codefactor.io/repository/github/cheginit/geomesher/overview/main
    :alt: CodeFactor

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

|

Features
--------

GeoMesher is a fork of `pandamesh <https://github.com/Deltares/pandamesh>`__. The original
package included two mesh generators: `Triangle <https://www.cs.cmu.edu/~quake/triangle.html>`__
and `Gmsh <https://gmsh.info/>`__. This fork only includes the Gmsh mesh generator since
Triangle seems to be not maintained anymore. Also, GeoMesher adds the following new
functionalities:

* A new method for returning the generated mesh as a GeoDataFrame.
* A new function called ``gdf_mesher`` that can generate a mesh from a GeoDataFrame
  with a single function call and with sane defaults for the mesh generator.
* Remap a scalar field from the source GeoDataFrame to the generated mesh,
  using an area weighted interpolation method
  (based on `Tobler <https://github.com/pysal/tobler>`__).
* Handle ``MultiPolygon`` geometries in the source GeoDataFrame.

Note that the remapping functionality of GeoMesher is a simple areal interpolation method.
For more advanced interpolation methods, please use `Tobler <https://pysal.org/tobler/index.html>`__.

Installation
------------

You can install GeoMesher using ``pip``:

.. code-block:: console

    $ pip install geomesher

or using ``conda`` (``mamba``):

.. code-block:: console

    $ conda install -c conda-forge geomesher

Quick start
-----------

The following example shows how to generate a mesh from a GeoDataFrame
using both the ``gdf_mesher`` function and the ``GmshMesher`` class.

We start by getting a GeoDataFrame of South America from the Natural Earth website.
Then, we reproject it to a projected coordinate system (UTM zone 20S).
Finally, we add a new column called ``cellsize`` that will be used to set the
maximum size of the mesh elements.

We use the ``gdf_mesher`` function to generate the mesh with default parameters
and use ``GmshMesher`` to generate the mesh with ``MESH_ADAPT`` algorithm.
We also use the ``area_interpolate`` function to remap the ``POP_EST`` column
from the source GeoDataFrame to the generated mesh.

.. code:: python

    import geopandas as gpd
    import geomesher as gm

    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )

    south_america = world[world["CONTINENT"] == "South America"]
    south_america = south_america.explode(ignore_index=True).to_crs(32620)
    south_america["cellsize"] = 500_000.0

    mesh_auto = gm.gdf_mesher(south_america, intensive_variables=["POP_EST"])

    mesher = gm.Mesher(south_america)
    mesher.mesh_algorithm = "MESH_ADAPT"
    mesh_adapt = mesher.generate()
    mesh_adapt = gm.area_interpolate(south_america, mesh_adapt, intensive_variables=["POP_EST"])

.. image:: https://raw.githubusercontent.com/cheginit/geomesher/main/doc/source/_static/demo.png
  :target: https://github.com/cheginit/geomesher

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pygeoogc/blob/main/CONTRIBUTING.rst>`__
file for instructions.

Credits
-------

GeoMesher is a fork of `pandamesh <https://github.com/Deltares/pandamesh>`__ (MIT License)
and uses one of the modules in
`Tobler <https://pysal.org/tobler/index.html>`__ (BSD-3-Clause License).
