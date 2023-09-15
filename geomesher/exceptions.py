"""Customized Hydrodata exceptions."""
from __future__ import annotations

from typing import Generator, Sequence


class MissingCRSError(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self, gdf: str) -> None:
        self.message = f"CRS of {gdf} is missing."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ProjectedCRSError(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self) -> None:
        self.message = "Input dataframes must be in a projected CRS."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MatchingCRSError(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self) -> None:
        self.message = "Input dataframes are in different CRS."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingColumnsError(Exception):
    """Exception raised when a required column is missing from a dataframe.

    Parameters
    ----------
    missing : list
        List of missing columns.
    """

    def __init__(self, missing: list[str]) -> None:
        self.message = "The following columns are missing:\n" + f"{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputValueError(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(
        self,
        inp: str,
        valid_inputs: Sequence[str | int] | Generator[str | int, None, None],
    ) -> None:
        self.message = f"Given {inp} is invalid. Valid options are:\n" + ", ".join(
            str(i) for i in valid_inputs
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputTypeError(TypeError):
    """Exception raised when a function argument type is invalid.

    Parameters
    ----------
    arg : str
        Name of the function argument
    valid_type : str
        The valid type of the argument
    example : str, optional
        An example of a valid form of the argument, defaults to None.
    """

    def __init__(self, arg: str, valid_type: str, example: str | None = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputRangeError(ValueError):
    """Exception raised when a function argument is not in the valid range."""
