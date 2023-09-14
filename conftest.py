"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pygeohydro namespace for doctest."""
    import geomesher as gm

    doctest_namespace["gm"] = gm
