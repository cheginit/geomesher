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

    mesher = gm.GmshMesher(south_america)
    mesher.mesh_algorithm = "MESH_ADAPT"
    mesh_adapt = mesher.generate_gdf()
    mesh_adapt = gm.area_interpolate(south_america, mesh_adapt, intensive_variables=["POP_EST"])

.. image:: https://raw.githubusercontent.com/cheginit/geomesher/main/doc/source/_static/demo.png
  :target: https://github.com/cheginit/geomesher
