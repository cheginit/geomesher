"""Area Weighted Interpolation based on `tobler <https://github.com/pysal/tobler>`__."""
from __future__ import annotations

from typing import Literal, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from scipy.sparse import coo_matrix, csr_matrix, diags

from geomesher.exceptions import (
    InputValueError,
    MatchingCRSError,
    MissingCRSError,
    ProjectedCRSError,
)

__all__ = ["area_interpolate"]


def _finite_check(df: gpd.GeoDataFrame, column: str) -> npt.NDArray[np.float64]:
    """Check if variable has nan or inf values and replace them with 0.0."""
    values = df[column].to_numpy("f8")
    values[~np.isfinite(values)] = 0.0
    return values


def _area_tables_binning(
    source_df: gpd.GeoDataFrame,
    target_df: gpd.GeoDataFrame,
    spatial_index: Literal["source", "target", "auto"],
) -> csr_matrix:
    """Construct area allocation and source-target correspondence tables using a spatial indexing approach.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        The source dataframe to get values from.
    target_df : geopandas.GeoDataFrame
        The target dataframe to interpolate the values from ``source_df``.
    spatial_index : str
        Spatial index to use to build the allocation of area from source
        to target tables. It currently supports
        the following values:

        - ``source``: build the spatial index on ``source_df``
        - ``target``: build the spatial index on ``target_df``
        - ``auto``: attempts to guess the most efficient alternative.
            Currently, this option uses the largest table to build the
            index, and performs a ``query`` on the shorter table.

    Returns
    -------
    tables : scipy.sparse.csr_matrix

    """
    df1 = source_df.copy()
    df2 = target_df.copy()

    # it is generally more performant to use the longer df as spatial index
    if spatial_index == "auto":
        spatial_index = "source" if df1.shape[0] > df2.shape[0] else "target"

    if spatial_index == "source":
        ids_tgt, ids_src = df1.sindex.query(df2.geometry, predicate="intersects")
    elif spatial_index == "target":
        ids_src, ids_tgt = df2.sindex.query(df1.geometry, predicate="intersects")
    else:
        raise InputValueError("spatial_index", ["source", "target", "auto"])

    areas = shapely.intersection(
        df1.geometry.iloc[ids_src].reset_index(drop=True),
        df2.geometry.iloc[ids_tgt].reset_index(drop=True),
    ).area.to_numpy("f4")

    table = coo_matrix(
        (
            areas,
            (ids_src, ids_tgt),
        ),
        shape=(df1.shape[0], df2.shape[0]),
        dtype="f4",
    )

    return table.tocsr()


def area_interpolate(
    source_df: gpd.GeoDataFrame,
    target_df: gpd.GeoDataFrame,
    extensive_variables: str | list[str] | None = None,
    intensive_variables: str | list[str] | None = None,
    categorical_variables: str | list[str] | None = None,
    table: csr_matrix | None = None,
    allocate_total: bool = True,
    spatial_index: Literal["source", "target", "auto"] = "auto",
) -> gpd.GeoDataFrame:
    r"""Area interpolation for extensive, intensive and categorical variables.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        The source dataframe to get values from.
    target_df : geopandas.GeoDataFrame
        The target dataframe to interpolate the values from ``source_df``.
    extensive_variables : list, optional
        Columns in dataframes for extensive variables, defaults to ``None``.
    intensive_variables : list, optional
        Columns in dataframes for intensive variables, defaults to ``None``.
    categorical_variables : list, optional
        Columns in dataframes for categorical variables, defaults to ``None``.
    table : scipy.sparse.csr_matrix, optional
        Area allocation source-target correspondence
        table. If not provided, it will be built from ``source_df`` and
        ``target_df``.
    allocate_total : boolean, optional
        True if total value of source area should be
        allocated. False if denominator is area of i. Note that the two cases
        would be identical when the area of the source polygon is exhausted by
        intersections. See Notes for more details. Defaults to True.
    spatial_index : str, optional
        Spatial index to use to build the allocation of area from source
        to target tables, defaults to ``auto``. It currently supports
        the following values:

        - ``source``: build the spatial index on ``source_df``
        - ``target``: build the spatial index on ``target_df``
        - ``auto``: attempts to guess the most efficient alternative.
            Currently, this option uses the largest table to build the
            index, and performs a ``query`` on the shorter table.

    Returns
    -------
    geopandas.GeoDataFrame
        new geodaraframe with interpolated variables as columns and ``target_df`` geometry
        as output geometry

    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.
    For an extensive variable, the estimate at target polygon :math:`j` (default case) is:

    .. math::

        v_j = \sum_i v_i w_{i,j}

        w_{i,j} = \frac{a_{i,j}}{\sum_k a_{i,k}}

    If the area of the source polygon is not exhausted by intersections with
    target polygons and there is reason to not allocate the complete value of
    an extensive attribute, then setting ``allocate_total=False`` will use the
    following weights:

    .. math::

        v_j = \sum_i v_i w_{i,j}

        w_{i,j} = \frac{a_{i,j}}{a_i}

    where :math:`a_i` is the total area of source polygon :math:`i`.
    For an intensive variable, the estimate at target polygon :math:`j` is:

    .. math::

        v_j = \sum_i v_i w_{i,j}

        w_{i,j} = \frac{a_{i,j}}{\sum_k a_{k,j}}

    For categorical variables, the estimate returns ratio of presence of each
    unique category.
    """
    if not source_df.crs:
        raise MissingCRSError("source_df")

    if not target_df.crs:
        raise MissingCRSError("target_df")

    if source_df.crs != target_df.crs:
        raise MatchingCRSError

    if not source_df.crs.is_projected:
        raise ProjectedCRSError

    source_df = source_df.copy()
    target_df = target_df.copy()

    if table is None:
        table = _area_tables_binning(source_df, target_df, spatial_index)

    dfs = []
    extensive = []
    if extensive_variables:
        extensive_variables = (
            [extensive_variables] if isinstance(extensive_variables, str) else extensive_variables
        )
        den = source_df.area.to_numpy("f8")
        if allocate_total:
            den = np.asarray(table.sum(axis=1), "f8")
        den = cast("npt.NDArray[np.float64]", den)
        den = den + (den == 0)
        den = 1.0 / den
        n = den.shape[0]
        den = den.reshape((n,))
        den = diags([den], [0])  # pyright: ignore[reportGeneralTypeIssues]
        weights = den.dot(table)  # row standardize table

        for variable in extensive_variables:
            vals = _finite_check(source_df, variable)
            estimates = diags([vals], [0]).dot(weights)  # pyright: ignore[reportGeneralTypeIssues]
            estimates = estimates.sum(axis=0)
            extensive.append(estimates.tolist()[0])

        extensive = np.asarray(extensive, dtype="f8")
        extensive = pd.DataFrame(extensive.T, columns=extensive_variables)

    intensive = []
    if intensive_variables:
        intensive_variables = (
            [intensive_variables] if isinstance(intensive_variables, str) else intensive_variables
        )
        area = np.asarray(table.sum(axis=0))
        den = cast("npt.NDArray[np.float64]", 1.0 / (area + (area == 0)))
        n, k = den.shape
        den = den.reshape((k,))
        den = diags([den], [0])  # pyright: ignore[reportGeneralTypeIssues]
        weights = table.dot(den)

        for variable in intensive_variables:
            vals = _finite_check(source_df, variable)
            n = vals.shape[0]
            vals = vals.reshape((n,))
            estimates = diags([vals], [0])  # pyright: ignore[reportGeneralTypeIssues]
            estimates = estimates.dot(weights).sum(axis=0)
            intensive.append(estimates.tolist()[0])

        intensive = np.asarray(intensive)
        intensive = pd.DataFrame(intensive.T, columns=intensive_variables)

    categorical = {}
    if categorical_variables:
        categorical_variables = (
            [categorical_variables]
            if isinstance(categorical_variables, str)
            else categorical_variables
        )
        for variable in categorical_variables:
            unique = source_df[variable].unique()
            for value in unique:
                mask = source_df[variable] == value
                _cat = np.asarray(
                    table[mask].sum(axis=0)  # pyright: ignore[reportGeneralTypeIssues]
                )
                categorical[f"{variable}_{value}"] = _cat[0]

        categorical = pd.DataFrame(categorical)
        categorical = categorical.div(target_df.area.to_numpy(), axis=0)

    if extensive_variables:
        dfs.append(extensive)
    if intensive_variables:
        dfs.append(intensive)
    if categorical_variables:
        dfs.append(categorical)

    df = pd.concat(dfs, axis=1)
    df["geometry"] = target_df[target_df.geometry.name].reset_index(drop=True)
    df = gpd.GeoDataFrame(df.replace(np.inf, np.nan), index=target_df.index)
    return df
