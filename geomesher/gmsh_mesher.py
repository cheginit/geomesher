"""Generate mesh using GMSH."""
from __future__ import annotations

import json
import pathlib
import tempfile
from contextlib import contextmanager
from enum import Enum, IntEnum
from typing import Generator, Literal

import geopandas as gpd
import gmsh
import numpy as np
import pandas as pd
import shapely

from geomesher.common import (
    FloatArray,
    InputValueError,
    IntArray,
    check_geodataframe,
    repr,
    separate,
)
from geomesher.gmsh_fields import (
    FIELDS,
    add_distance_field,
    add_structured_field,
    validate_field,
)
from geomesher.gmsh_geometry import add_field_geometry, add_geometry

__all__ = [
    "FieldCombination",
    "GmshMesher",
    "MeshAlgorithm",
    "SubdivisionAlgorithm",
    "gmsh_env",
    "gdf_mesher",
]


@contextmanager
def gmsh_env() -> Generator[None, None, None]:
    if gmsh.isInitialized():
        gmsh.finalize()

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 0)
        yield
    finally:
        gmsh.finalize()


class MeshAlgorithm(IntEnum):
    """Each algorithm has its own advantages and disadvantages.

    For all 2D unstructured algorithms a Delaunay mesh that contains all
    the points of the 1D mesh is initially constructed using a
    divide-and-conquer algorithm. Missing edges are recovered using edge
    swaps. After this initial step several algorithms can be applied to
    generate the final mesh:

    * The MeshAdapt algorithm is based on local mesh modifications. This
      technique makes use of edge swaps, splits, and collapses: long edges
      are split, short edges are collapsed, and edges are swapped if a
      better geometrical configuration is obtained.
    * The Delaunay algorithm is inspired by the work of the GAMMA team at
      INRIA. New points are inserted sequentially at the circumcenter of
      the element that has the largest adimensional circumradius. The mesh
      is then reconnected using an anisotropic Delaunay criterion.
    * The Frontal-Delaunay algorithm is inspired by the work of S. Rebay.
    * Other experimental algorithms with specific features are also
      available. In particular, Frontal-Delaunay for Quads is a variant of
      the Frontal-Delaunay algorithm aiming at generating right-angle
      triangles suitable for recombination; and BAMG allows to generate
      anisotropic triangulations.

    For very complex curved surfaces the MeshAdapt algorithm is the most robust.
    When high element quality is important, the Frontal-Delaunay algorithm should
    be tried. For very large meshes of plane surfaces the Delaunay algorithm is
    the fastest; it usually also handles complex mesh size fields better than the
    Frontal-Delaunay. When the Delaunay or Frontal-Delaunay algorithms fail,
    MeshAdapt is automatically triggered. The Automatic algorithm uses
    Delaunay for plane surfaces and MeshAdapt for all other surfaces.
    """

    MESH_ADAPT = 1
    AUTOMATIC = 2
    INITIAL_MESH_ONLY = 3
    FRONTAL_DELAUNAY = 5
    BAMG = 7
    FRONTAL_DELAUNAY_FOR_QUADS = 8
    PACKING_OF_PARALLELLOGRAMS = 9


class SubdivisionAlgorithm(IntEnum):
    """All meshes can be subdivided to generate fully quadrangular cells."""

    NONE = 0
    ALL_QUADRANGLES = 1
    BARYCENTRIC = 3


class FieldCombination(Enum):
    """Controls how cell size fields are combined when they are found at the same location."""

    MIN = "Min"
    MAX = "Max"
    MEAN = "Mean"


def coerce_field(field: dict[str, str] | str) -> dict[str, str]:
    if not isinstance(field, (dict, str)):
        raise TypeError("field must be a dictionary or a valid JSON dictionary string")
    if isinstance(field, str):
        return json.loads(field)
    return field


class GmshMesher:
    """Wrapper for the python bindings to Gmsh.

    This class must be initialized
    with a ``geopandas.GeoDataFrame`` containing at least one polygon, and a column
    named ``"cellsize"``.

    Optionally, multiple polygons with different cell sizes can be included in
    the geodataframe. These can be used to achieve local mesh remfinement.

    Linestrings and points may also be included. The segments of linestrings
    will be directly forced into the triangulation. Points can also be forced
    into the triangulation. Unlike Triangle, the cell size values associated
    with these geometries **will** be used.

    Gmsh cannot automatically resolve overlapping polygons, or points
    located exactly on segments. During initialization, the geometries of
    the geodataframe are checked:

        * Polygons should not have any overlap with each other.
        * Linestrings should not intersect each other.
        * Every linestring should be fully contained by a single polygon;
          a linestring may not intersect two or more polygons.
        * Linestrings and points should not "touch" / be located on
          polygon borders.
        * Holes in polygons are fully supported, but they must not contain
          any linestrings or points.

    If such cases are detected, the initialization will error.

    For more details on Gmsh, see:
    https://gmsh.info/doc/texinfo/gmsh.html

    A helpful index can be found near the bottom:
    https://gmsh.info/doc/texinfo/gmsh.html#Syntax-index
    """

    @staticmethod
    def _initialize_gmsh() -> None:
        if gmsh.isInitialized():
            gmsh.finalize()

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 0)

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        self._initialize_gmsh()
        self.gdf_crs = gdf.crs
        check_geodataframe(gdf)
        polygons, linestrings, points = separate(gdf)

        # Include geometry into gmsh
        add_geometry(polygons, linestrings, points)

        # Initialize fields parameters
        self._current_field_id = 0
        self._fields_list: list[int] = []
        self._distance_fields_list: list[int] = []
        self.fields = gpd.GeoDataFrame()
        self._tmpdir = tempfile.TemporaryDirectory()

        # Set default values for meshing parameters
        self._mesh_algorithm = "AUTOMATIC"
        self.recombine_all = False
        self.force_geometry = True
        self.mesh_size_extend_from_boundary = True
        self.mesh_size_from_points = True
        self.mesh_size_from_curvature = False
        self._field_combination = "MIN"
        self._subdivision_algorithm = "NONE"

    @staticmethod
    def finalize_gmsh() -> None:
        """Finalize Gmsh."""
        gmsh.finalize()

    # Properties
    # ----------

    @property
    def mesh_algorithm(
        self,
    ) -> Literal[
        "MESH_ADAPT",
        "AUTOMATIC",
        "INITIAL_MESH_ONLY",
        "FRONTAL_DELAUNAY",
        "BAMG",
        "FRONTAL_DELAUNAY_FOR_QUADS",
        "PACKING_OF_PARALLELLOGRAMS",
    ]:
        """Can be set to one of :py:class:`geomesher.MeshAlgorithm`.

        .. code::

            MESH_ADAPT = 1
            AUTOMATIC = 2
            INITIAL_MESH_ONLY = 3
            FRONTAL_DELAUNAY = 5
            BAMG = 7
            FRONTAL_DELAUNAY_FOR_QUADS = 8
            PACKING_OF_PARALLELLOGRAMS = 9

        Each algorithm has its own advantages and disadvantages.
        """
        return self._mesh_algorithm

    @mesh_algorithm.setter
    def mesh_algorithm(
        self,
        value: Literal[
            "MESH_ADAPT",
            "AUTOMATIC",
            "INITIAL_MESH_ONLY",
            "FRONTAL_DELAUNAY",
            "BAMG",
            "FRONTAL_DELAUNAY_FOR_QUADS",
            "PACKING_OF_PARALLELLOGRAMS",
        ],
    ) -> None:
        if value not in MeshAlgorithm._member_names_:
            raise InputValueError("mesh_algorithm", MeshAlgorithm._member_names_)
        self._mesh_algorithm = value
        gmsh.option.setNumber("Mesh.Algorithm", MeshAlgorithm[value].value)

    @property
    def recombine_all(self) -> bool:
        """Apply recombination algorithm to all surfaces, ignoring per-surface spec."""
        return self._recombine_all

    @recombine_all.setter
    def recombine_all(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("recombine_all must be a bool")
        self._recombine_all = value
        gmsh.option.setNumber("Mesh.RecombineAll", value)

    @property
    def force_geometry(self) -> bool:
        """Force the geometry to be used in the mesh, or not per surface."""
        return self._force_geometry

    @force_geometry.setter
    def force_geometry(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("force_geometry must be a bool")
        self._force_geometry = value

    @property
    def mesh_size_extend_from_boundary(self) -> bool:
        """Forces the mesh size to be extended from the boundary, or not per surface."""
        return self._mesh_size_extend_from_boundary

    @mesh_size_extend_from_boundary.setter
    def mesh_size_extend_from_boundary(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("mesh_size_extend_from_boundary must be a bool")
        self._mesh_size_extend_from_boundary = value
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", value)

    @property
    def mesh_size_from_points(self) -> bool:
        """Compute mesh element sizes from values given at geometry points."""
        return self._mesh_size_from_points

    @mesh_size_from_points.setter
    def mesh_size_from_points(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("mesh_size_from_points must be a bool")
        self._mesh_size_from_points = value
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", value)

    @property
    def mesh_size_from_curvature(self) -> bool:
        """Automatically compute mesh element sizes from curvature.

        It uses the value as the target number of elements per 2 * Pi radians.
        """
        return self._mesh_size_from_curvature

    @mesh_size_from_curvature.setter
    def mesh_size_from_curvature(self, value: bool) -> None:
        """Automatically compute mesh element sizes from curvature.

        It uses the value as the target number of elements per 2 * Pi radians.
        """
        if not isinstance(value, bool):
            raise TypeError("mesh_size_from_curvature must be a bool")
        self._mesh_size_from_curvature = value
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", value)

    @property
    def field_combination(self) -> Literal["MIN", "MAX", "MEAN"]:
        """Control how cell size fields are combined.

        When they are found at the same location. Can be set to one of
        :py:class:`geomesher.FieldCombination`:

        .. code::

            MIN = "Min"
            MAX = "Max"
            MEAN = "Mean"

        """
        return self._field_combination

    @field_combination.setter
    def field_combination(self, value: Literal["MIN", "MAX", "MEAN"]) -> None:
        if value not in FieldCombination._member_names_:
            raise InputValueError("field_combination", FieldCombination._member_names_)
        self._field_combination = value

    @property
    def subdivision_algorithm(self) -> Literal["NONE", "ALL_QUADRANGLES", "BARYCENTRIC"]:
        """Subdivision algorithm.

        All meshes can be subdivided to generate fully quadrangular cells. Can
        be set to one of :py:class:`geomesher.SubdivisionAlgorithm`:

        .. code::

            NONE = 0
            ALL_QUADRANGLES = 1
            BARYCENTRIC = 3

        """
        return self._subdivision_algorithm

    @subdivision_algorithm.setter
    def subdivision_algorithm(
        self, value: Literal["NONE", "ALL_QUADRANGLES", "BARYCENTRIC"]
    ) -> None:
        if value not in SubdivisionAlgorithm._member_names_:
            raise InputValueError("subdivision_algorithm", SubdivisionAlgorithm._member_names_)
        self._subdivision_algorithm = value
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", SubdivisionAlgorithm[value].value)

    # Methods
    # -------
    def _new_field_id(self) -> int:
        self._current_field_id += 1
        return self._current_field_id

    def _combine_fields(self) -> None:
        # Create a combination field
        if self.fields is None:
            return
        field_id = self._new_field_id()
        gmsh.model.mesh.field.add(FieldCombination[self.field_combination].value, field_id)
        gmsh.model.mesh.field.setNumbers(field_id, "FieldsList", self._fields_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

    def clear_fields(self) -> None:
        """Clear all cell size fields from the mesher."""
        self.fields = None
        for field_id in self._fields_list + self._distance_fields_list:
            gmsh.model.mesh.field.remove(field_id)
        self._fields_list = []
        self._distance_fields_list = []
        self._current_field_id = 0

    def add_distance_field(self, gdf: gpd.GeoDataFrame, minimum_cellsize: float) -> None:
        """Add a distance field to the mesher.

        The of geometry of these fields are not forced into the mesh, but they
        can be used to specify zones of with cell sizes.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Location and cell size of the fields, as vector data.
        minimum_cellsize: float
            Minimum cell size.
        """
        if "field" not in gdf.columns:
            raise ValueError("field column is missing from geodataframe")

        # Explode just in case to get rid of multi-elements
        gdf = gdf.explode()

        for field, field_gdf in gdf.groupby("field"):
            distance_id = self._new_field_id()
            field_id = self._new_field_id()
            field_dict = coerce_field(field)
            fieldtype = field_dict["type"].lower()

            spec, add_field = FIELDS[fieldtype.lower()]
            try:
                validate_field(field_dict, spec)
            except KeyError as ex:
                raise ValueError(
                    f'invalid field type {fieldtype}. Allowed are: "MathEval", "Threshold".'
                ) from ex

            # Insert geometry, and create distance field
            nodes_list = add_field_geometry(field_gdf, minimum_cellsize)
            add_distance_field(nodes_list, [], 0, distance_id)

            # Add field based on distance field
            add_field(field_dict, distance_id=distance_id, field_id=field_id)
            self._fields_list.append(field_id)
            self._distance_fields_list.append(distance_id)

        self.fields = pd.concat(self.fields, gdf)

    def add_structured_field(
        self,
        cellsize: FloatArray,
        xmin: float,
        ymin: float,
        dx: float,
        dy: float,
        outside_value: float | None = None,
    ) -> None:
        """Add a structured field specifying cell sizes.

        Gmsh will interpolate between the points to determine the desired cell size.

        Parameters
        ----------
        cellsize: FloatArray with shape ``(n_y, n_x)``
            Specifies the cell size on a structured grid. The location of this grid
            is determined by ``xmin, ymin, dx, dy``.
        xmin: float
            x-origin.
        ymin: float
            y-origin.
        dx: float
            Spacing along the x-axis.
        dy: float
            Spacing along the y-axis.
        outside_value: float, optional
            Value outside of the window ``(xmin, xmax)`` and ``(ymin, ymax)``.
            Default value is None.
        """
        if outside_value is not None:
            set_outside_value = True
        else:
            set_outside_value = False
            outside_value = -1.0

        field_id = self._new_field_id()
        path = f"{self._tmpdir.name}/structured_field_{field_id}.dat"
        add_structured_field(
            cellsize,
            xmin,
            ymin,
            dx,
            dy,
            outside_value,
            set_outside_value,
            field_id,
            path,
        )
        self._fields_list.append(field_id)

    def _vertices(self) -> FloatArray:
        # getNodes returns: node_tags, coord, parametric_coord
        _, vertices, _ = gmsh.model.mesh.getNodes()
        return vertices.reshape((-1, 3))[:, :2]

    def _faces(self) -> IntArray:
        element_types, _, node_tags = gmsh.model.mesh.getElements()
        tags = dict(zip(element_types, node_tags))
        _triangle = 2
        _quad = 3
        _fill_value = 0
        # Combine triangle and quad faces if the mesh is heterogeneous
        if _triangle in tags and _quad in tags:
            triangle_faces = tags[_triangle].reshape((-1, 3))
            quad_faces = tags[_quad].reshape((-1, 4))
            n_triangle = triangle_faces.shape[0]
            n_quad = quad_faces.shape[0]
            faces = np.full((n_triangle + n_quad, 4), _fill_value)
            faces[:n_triangle, :3] = triangle_faces
            faces[n_triangle:, :] = quad_faces
        elif _quad in tags:
            faces = tags[_quad].reshape((-1, 4))
        elif _triangle in tags:
            faces = tags[_triangle].reshape((-1, 3))
        else:
            raise ValueError("No triangles or quads in mesh")
        # convert to 0-based index
        return faces - 1

    def generate(self) -> tuple[FloatArray, IntArray]:
        """
        Generate a mesh of triangles or quadrangles.

        Returns
        -------
        vertices: np.ndarray of floats with shape ``(n_vertex, 2)``
        faces: np.ndarray of integers with shape ``(n_face, nmax_per_face)``
            ``nmax_per_face`` is 3 for exclusively triangles and 4 if
            quadrangles are included. A fill value of -1 is used as a last
            entry for triangles in that case.
        """
        self._combine_fields()
        gmsh.model.mesh.generate(dim=2)

        # cleaning up of mesh in order to obtain unique elements and nodes
        gmsh.model.mesh.removeDuplicateElements()
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.renumberElements()
        gmsh.model.mesh.renumberNodes()

        return self._vertices(), self._faces()

    def generate_gdf(self) -> gpd.GeoSeries:
        """Generate the mesh and return it as a ``geopandas.GeoSeries``."""
        vertices, faces = self.generate()
        nodes = vertices[faces]
        return gpd.GeoSeries(shapely.polygons(nodes), crs=self.gdf_crs)

    def write(self, path: str | pathlib.Path):
        """Write a gmsh .msh file.

        Parameters
        ----------
        path: str or pathlib.Path
        """
        gmsh.write(path)

    def __repr__(self):
        return repr(self)


def gdf_mesher(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Generate a mesh from a geodataframe.

    This function uses defaults Gmsh parameters. For more control, use
    :py:class:`geomesher.GmshMesher`.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        The input must have at least one polygon and a column named ``cellsize``.
        Optionally, multiple polygons with different cell sizes can be included in
        the geodataframe. These can be used to achieve local mesh remfinement.

    Returns
    -------
    mesh: geopandas.GeoSeries
        The mesh as a geopandas.GeoSeries.

    Notes
    -----
    Linestrings and points may also be included. The segments of linestrings
    will be directly forced into the triangulation. Points can also be forced
    into the triangulation. Unlike Triangle, the cell size values associated
    with these geometries **will** be used.

    Gmsh cannot automatically resolve overlapping polygons, or points
    located exactly on segments. During initialization, the geometries of
    the geodataframe are checked:

        * Polygons should not have any overlap with each other.
        * Linestrings should not intersect each other.
        * Every linestring should be fully contained by a single polygon;
          a linestring may not intersect two or more polygons.
        * Linestrings and points should not "touch" / be located on
          polygon borders.
        * Holes in polygons are fully supported, but they must not contain
          any linestrings or points.

    If such cases are detected, the initialization will error.

    For more details on Gmsh, see:
    https://gmsh.info/doc/texinfo/gmsh.html

    A helpful index can be found near the bottom:
    https://gmsh.info/doc/texinfo/gmsh.html#Syntax-index
    """
    mesher = GmshMesher(gdf)
    mesh = mesher.generate_gdf()
    mesher.finalize_gmsh()
    return mesh
