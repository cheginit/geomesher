"""Top-level API for geomesher."""
from importlib.metadata import PackageNotFoundError, version

from geomesher import exceptions
from geomesher.area_weighted import area_interpolate
from geomesher.mesher import (
    Mesher,
    gdf_mesher,
    gmsh_env,
)

try:
    __version__ = version("geomesher")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "Mesher",
    "gdf_mesher",
    "gmsh_env",
    "area_interpolate",
    "exceptions",
]
