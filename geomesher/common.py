"""Some helper functions."""
from __future__ import annotations

import functools
import operator
from itertools import combinations
from typing import TYPE_CHECKING, Any, Sequence, TypeVar, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from pandas.api.types import is_integer_dtype

from geomesher.exceptions import InputTypeError

if TYPE_CHECKING:
    from shapely import MultiPolygon, Polygon

    GDFTYPE = TypeVar("GDFTYPE", gpd.GeoDataFrame, gpd.GeoSeries)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
coord_dtype = np.dtype([("x", np.float64), ("y", np.float64)])


def repr(obj: Any) -> str:
    strings = [type(obj).__name__]
    for k, v in obj.__dict__.items():
        if k.startswith("_"):
            k = k[1:]
        if isinstance(v, np.ndarray):
            s = f"    {k} = np.ndarray with shape({v.shape})"
        else:
            s = f"    {k} = {v}"
        strings.append(s)
    return "\n".join(strings)


def flatten(seq: Sequence[Any]):
    return functools.reduce(operator.concat, seq)


def check_geodataframe(features: gpd.GeoDataFrame) -> None:
    if not isinstance(features, gpd.GeoDataFrame):
        raise TypeError(f"Expected GeoDataFrame, received instead: {type(features).__name__}")
    if "cellsize" not in features:
        colnames = list(features.columns)
        raise ValueError(f'Missing column "cellsize" in columns: {colnames}')
    if len(features) == 0:
        raise ValueError("Dataframe is empty")
    if not is_integer_dtype(features.index):
        raise ValueError(
            f"geodataframe index is not integer typed, received: {features.index.dtype}"
        )
    if features.index.duplicated().any():
        raise ValueError("geodataframe index contains duplicates")


def overlap_shortlist(features: gpd.GeoSeries) -> tuple[IntArray, IntArray]:
    """Create a shortlist of polygons or linestrings indices."""
    bounds = features.bounds
    index_a, index_b = (np.array(index) for index in zip(*combinations(features.index, 2)))
    df_a = bounds.loc[index_a]
    df_b = bounds.loc[index_b]
    # Convert to dict to get rid of clashing index.
    a = {k: df_a[k].to_numpy("f8") for k in df_a}
    b = {k: df_b[k].to_numpy("f8") for k in df_b}
    # Touching does not count as overlap here.
    overlap = (
        (a["maxx"] >= b["minx"])
        & (b["maxx"] >= a["minx"])
        & (a["maxy"] >= b["miny"])
        & (b["maxy"] >= a["miny"])
    )
    return index_a[overlap], index_b[overlap]


def intersecting_features(features: gpd.GeoSeries, feature_type: str) -> tuple[IntArray, IntArray]:
    # Check all combinations where bounding boxes overlap.
    index_a, index_b = overlap_shortlist(features)
    unique = np.unique(np.concatenate([index_a, index_b]))

    # Now do the expensive intersection check.
    # Polygons that touch are allowed, but they result in intersects() == True.
    # To avoid this, we create temporary geometries that are slightly smaller
    # by buffering with a small negative value.
    shortlist = features.loc[unique]
    if feature_type == "polygon":
        shortlist = shortlist.buffer(-1.0e-6)
    a = shortlist.loc[index_a]
    b = shortlist.loc[index_b]
    # Synchronize index so there's a one to one (row to row) intersection
    # check.
    a.index = np.arange(len(a))
    b.index = np.arange(len(b))
    with_overlap = a.intersects(b).values
    return index_a[with_overlap], index_b[with_overlap]


def check_intersection(features: gpd.GeoSeries, feature_type: str) -> None:
    index_a, index_b = intersecting_features(features, feature_type)
    n_overlap = len(index_a)
    if n_overlap > 0:
        message = "\n".join([f"{a} with {b}" for a, b, in zip(index_a, index_b)])
        raise ValueError(f"{n_overlap} cases of intersecting {feature_type} detected:\n{message}")


def check_features(features: gpd.GeoSeries, feature_type: str) -> None:
    """Check whether features are valid.

    Features should:

        * be simple: no self-intersection
        * not intersect with other features

    """
    # Check valid
    are_simple = features.is_simple
    n_complex = (~are_simple).sum()
    if n_complex > 0:
        raise ValueError(
            f"{n_complex} cases of complex {feature_type} detected: these "
            " features contain self intersections"
        )

    if len(features) <= 1:
        return

    check_intersection(features, feature_type)


def check_polygons(polygons: gpd.GeoSeries) -> None:
    check_features(polygons, "polygon")


def check_linestrings(
    linestrings: gpd.GeoSeries,
    polygons: gpd.GeoSeries,
) -> None:
    """Check whether linestrings are fully contained in a single polygon."""
    check_features(linestrings, "linestring")

    intersects = gpd.GeoDataFrame(geometry=linestrings).sjoin(
        gpd.GeoDataFrame(geometry=polygons),
        predicate="within",
    )
    n_diff = len(linestrings) - len(intersects)
    if n_diff != 0:
        raise ValueError(
            "The same linestring detected in multiple polygons or "
            "linestring detected outside of any polygon; "
            "a linestring must be fully contained by a single polygon."
        )


def check_points(
    points: gpd.GeoSeries,
    polygons: gpd.GeoSeries,
) -> None:
    """Check whether points are contained by a polygon."""
    within = gpd.GeoDataFrame(geometry=points).sjoin(
        gpd.GeoDataFrame(geometry=polygons),
        predicate="within",
    )
    n_outside = len(points) - len(within)
    if n_outside != 0:
        raise ValueError(f"{n_outside} points detected outside of a polygon")


def _get_area_range(mp: MultiPolygon) -> float:
    """Get the range of areas of polygons in a multipolygon."""
    if np.isclose(mp.area, 0.0):
        return 0.0
    return np.ptp([g.area for g in mp.geoms]) / mp.area


def _get_larges(mp: MultiPolygon) -> Polygon:
    """Get the largest polygon from a multipolygon."""
    return Polygon(
        mp.geoms[
            np.argmax([g.area for g in mp.geoms])
        ].exterior  # pyright: ignore[reportOptionalMemberAccess]
    )


def multi_explode(gdf: GDFTYPE) -> GDFTYPE:
    """Convert multi-part geometries to single-part and fill polygon holes, if any.

    Notes
    -----
    This function tries to convert multi-geometries to their constituents by
    first checking if multiploygons can be directly converted using
    their exterior boundaries. If not, it will try to remove those small
    sub-polygons that their area is less than 1% of the total area
    of the multipolygon. If this fails, the multi-geometries will be exploded.
    Thus, the number of rows in the output GeoDataFrame may be larger than
    the input GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with single-part geometries.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with polygons.
    """
    if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InputTypeError("gdf", "GeoDataFrame or GeoSeries")

    gdf_prj = cast("GDFTYPE", gdf.copy())
    if isinstance(gdf_prj, gpd.GeoSeries):
        gdf_prj = gpd.GeoDataFrame(gdf_prj.to_frame("geometry"))

    mp_idx = gdf_prj.loc[gdf_prj.geom_type == "MultiPolygon"].index
    if mp_idx.size > 0:
        geo_mp = gdf_prj.loc[mp_idx, "geometry"]
        geo_mp = cast("gpd.GeoSeries", geo_mp)
        idx = {i: g.geoms[0] for i, g in geo_mp.geometry.items() if len(g.geoms) == 1}
        gdf_prj.loc[list(idx), "geometry"] = list(idx.values())
        if len(idx) < len(geo_mp):
            area_rng = geo_mp.map(_get_area_range)
            mp_idx = area_rng[area_rng >= 0.99].index  # pyright: ignore[reportGeneralTypeIssues]
            if mp_idx.size > 0:
                gdf_prj.loc[mp_idx, "geometry"] = geo_mp.map(_get_larges)

    if gdf_prj.geom_type.str.contains("Multi").any():
        gdf_prj["multipart"] = [
            list(g.geoms) if "Multi" in g.type else [g] for g in gdf_prj.geometry
        ]
        gdf_prj = gdf_prj.explode("multipart")
        gdf_prj = cast("gpd.GeoDataFrame", gdf_prj)
        gdf_prj = gdf_prj.set_geometry("multipart", drop=True)
        gdf_prj = cast("gpd.GeoDataFrame", gdf_prj)

    if not gdf_prj.is_simple.all():
        gdf_prj["geometry"] = gdf_prj.buffer(0)

    if isinstance(gdf, gpd.GeoSeries):
        return gdf_prj.geometry

    return gdf_prj


def separate(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    geom_type = gdf.geom_type
    acceptable = ["Polygon", "LineString", "Point"]
    gdf = multi_explode(gdf)
    if not geom_type.isin(acceptable).all():
        raise TypeError(f"Geometry should be one of {acceptable}")

    polygons = gdf.loc[geom_type == "Polygon"].copy()
    polygons = cast("gpd.GeoDataFrame", polygons)
    linestrings = gdf.loc[geom_type == "LineString"].copy()
    linestrings = cast("gpd.GeoDataFrame", linestrings)
    points = gdf.loc[geom_type == "Point"].copy()
    points = cast("gpd.GeoDataFrame", points)
    for df in (polygons, linestrings, points):
        df["cellsize"] = df.cellsize.astype(np.float64)
        df.crs = None

    check_polygons(polygons.geometry)
    check_linestrings(linestrings.geometry, polygons.geometry)
    check_points(points.geometry, polygons.geometry)

    return polygons, linestrings, points
