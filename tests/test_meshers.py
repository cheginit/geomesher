import geopandas as gpd
import numpy as np
import shapely.geometry as sg

import geomesher as pm

outer_coords = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
inner_coords = np.array([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
line_coords = np.array([(2.0, 8.0), (8.0, 2.0)])
inner = sg.LinearRing(inner_coords)
outer = sg.LinearRing(outer_coords)
line = sg.LineString(line_coords)
donut = sg.Polygon(outer, holes=[inner])
refined = sg.Polygon(inner_coords)


def area(vertices, triangles):
    """Compute the area of every triangle in the mesh. (Helper for these tests.)."""
    coords = vertices[triangles]
    u = coords[:, 1] - coords[:, 0]
    v = coords[:, 2] - coords[:, 0]
    return 0.5 * np.abs(np.cross(u, v))


def gmsh_generate(gdf: gpd.GeoDataFrame):
    mesher = pm.GmshMesher(gdf)
    return mesher.generate()


def test_basic():
    polygon = sg.Polygon(outer)
    gdf = gpd.GeoDataFrame(geometry=[polygon])
    gdf["cellsize"] = 1.0
    vertices, triangles = gmsh_generate(gdf)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, polygon.area)


def test_hole():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    vertices, triangles = gmsh_generate(gdf)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, donut.area)


def test_adjacent_donut():
    inner_coords2 = inner_coords.copy()
    outer_coords2 = outer_coords.copy()
    inner_coords2[:, 0] += 10.0
    outer_coords2[:, 0] += 10.0
    inner2 = sg.LinearRing(inner_coords2)
    outer2 = sg.LinearRing(outer_coords2)
    donut2 = sg.Polygon(outer2, holes=[inner2])

    gdf = gpd.GeoDataFrame(geometry=[donut, donut2])
    gdf["cellsize"] = [1.0, 0.5]
    vertices, triangles = gmsh_generate(gdf)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, 2 * donut.area)

    # With a line at y=8.0 and points in the left polygon, at y=2.0
    line1 = sg.LineString([(0.25, 8.0), (9.75, 8.0)])
    line2 = sg.LineString([(10.25, 8.0), (19.75, 8.0)])
    x = np.arange(0.25, 10.0, 0.25)
    y = np.full(x.size, 2.0)
    points = gpd.points_from_xy(x=x, y=y)
    gdf = gpd.GeoDataFrame(geometry=[donut, donut2, line1, line2, *points])
    gdf["cellsize"] = 1.0

    vertices, triangles = gmsh_generate(gdf)
    mesh_area = area(vertices, triangles).sum()
    assert np.allclose(mesh_area, 2 * donut.area)


def test_gmsh_properties():
    gdf = gpd.GeoDataFrame(geometry=[donut])
    gdf["cellsize"] = 1.0
    mesher = pm.GmshMesher(gdf)

    # Set default values for meshing parameters
    mesher.mesh_algorithm = "FRONTAL_DELAUNAY"
    mesher.recombine_all = False
    mesher.force_geometry = True
    mesher.mesh_size_extend_from_boundary = False
    mesher.mesh_size_from_points = False
    mesher.mesh_size_from_curvature = True
    mesher.field_combination = "MEAN"
    mesher.subdivision_algorithm = "BARYCENTRIC"

    assert mesher.mesh_algorithm == "FRONTAL_DELAUNAY"
    assert mesher.recombine_all is False
    assert mesher.force_geometry is True
    assert mesher.mesh_size_extend_from_boundary is False
    assert mesher.mesh_size_from_points is False
    assert mesher.mesh_size_from_curvature is True
    assert mesher.field_combination == "MEAN"
    assert mesher.subdivision_algorithm == "BARYCENTRIC"
