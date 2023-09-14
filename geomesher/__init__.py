"""Top-level API for geomesher."""
from importlib.metadata import PackageNotFoundError, version

from geomesher.gmsh_mesher import (
    FieldCombination,
    GmshMesher,
    MeshAlgorithm,
    SubdivisionAlgorithm,
    gdf_mesher,
    gmsh_env,
)

try:
    __version__ = version("geomesher")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "FieldCombination",
    "GmshMesher",
    "MeshAlgorithm",
    "SubdivisionAlgorithm",
    "gdf_mesher",
    "gmsh_env",
]
