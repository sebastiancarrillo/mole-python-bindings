"""Tests for MOLE interpolation operators exposed to Python."""

import numpy as np
import pytest
from scipy.sparse import issparse

import _mole


def assert_valid_interpolation(operator):
    """Verify that an interpolation operator is usable."""

    assert issparse(operator)
    assert operator.ndim == 2
    assert operator.shape[0] > 0
    assert operator.shape[1] > 0
    assert operator.nnz > 0
    assert np.all(np.isfinite(operator.data))

    # Verify matrix-vector multiplication
    input_vector = np.ones(operator.shape[1])
    output_vector = operator @ input_vector

    assert output_vector.shape == (operator.shape[0],)
    assert np.all(np.isfinite(output_vector))


# Standard interpolation operators
@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        (
            "interpolation_1d",
            (8, 0.5),
        ),
        (
            "interpolation_2d",
            (8, 6, 0.5, 0.5),
        ),
        (
            "interpolation_3d",
            (6, 5, 4, 0.5, 0.5, 0.5),
        ),
    ],
)
def test_standard_interpolation(function_name, arguments):
    function = getattr(_mole, function_name)
    operator = function(*arguments)

    assert_valid_interpolation(operator)


# Alternate interpolation operators
@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        (
            "interpolation_alternate_1d",
            (True, 8, 0.5),
        ),
        (
            "interpolation_alternate_1d",
            (False, 8, 0.5),
        ),
        (
            "interpolation_alternate_2d",
            (True, 8, 6, 0.5, 0.5),
        ),
        (
            "interpolation_alternate_2d",
            (False, 8, 6, 0.5, 0.5),
        ),
        (
            "interpolation_alternate_3d",
            (True, 6, 5, 4, 0.5, 0.5, 0.5),
        ),
        (
            "interpolation_alternate_3d",
            (False, 6, 5, 4, 0.5, 0.5, 0.5),
        ),
    ],
)
def test_alternate_interpolation(function_name, arguments):
    function = getattr(_mole, function_name)
    operator = function(*arguments)

    assert_valid_interpolation(operator)


# Boundary-condition coefficient arrays
DIRICHLET_1D = np.array([1, 1], dtype=np.int64)
NEUMANN_1D = np.array([0, 0], dtype=np.int64)

DIRICHLET_2D = np.array([1, 1, 1, 1], dtype=np.int64)
NEUMANN_2D = np.array([0, 0, 0, 0], dtype=np.int64)

DIRICHLET_3D = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
NEUMANN_3D = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)


# Specialized cell/face/node interpolation operators
@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        # 1D
        (
            "interpolate_cell_to_face_1d",
            (2, 8, DIRICHLET_1D, NEUMANN_1D),
        ),
        (
            "interpolate_face_to_cell_1d",
            (2, 8, DIRICHLET_1D, NEUMANN_1D),
        ),
        (
            "interpolate_cell_to_node_1d",
            (2, 8, DIRICHLET_1D, NEUMANN_1D),
        ),
        (
            "interpolate_node_to_cell_1d",
            (2, 8, DIRICHLET_1D, NEUMANN_1D),
        ),

        # 2D
        (
            "interpolate_cell_to_face_2d",
            (2, 8, 6, DIRICHLET_2D, NEUMANN_2D),
        ),
        (
            "interpolate_face_to_cell_2d",
            (2, 8, 6, DIRICHLET_2D, NEUMANN_2D),
        ),
        (
            "interpolate_cell_to_node_2d",
            (2, 8, 6, DIRICHLET_2D, NEUMANN_2D),
        ),
        (
            "interpolate_node_to_cell_2d",
            (2, 8, 6, DIRICHLET_2D, NEUMANN_2D),
        ),

        # 3D
        (
            "interpolate_cell_to_face_3d",
            (2, 6, 5, 5, DIRICHLET_3D, NEUMANN_3D),
        ),
        (
            "interpolate_face_to_cell_3d",
            (2, 6, 5, 5, DIRICHLET_3D, NEUMANN_3D),
        ),
        (
            "interpolate_cell_to_node_3d",
            (2, 6, 5, 5, DIRICHLET_3D, NEUMANN_3D),
        ),
        (
            "interpolate_node_to_cell_3d",
            (2, 6, 5, 5, DIRICHLET_3D, NEUMANN_3D),
        ),
    ],
)
def test_specialized_interpolation(function_name, arguments):
    function = getattr(_mole, function_name)
    operator = function(*arguments)

    assert_valid_interpolation(operator)


EXPECTED_FUNCTIONS = [
    "interpolation_1d",
    "interpolation_2d",
    "interpolation_3d",
    "interpolation_alternate_1d",
    "interpolation_alternate_2d",
    "interpolation_alternate_3d",
    "interpolate_cell_to_face_1d",
    "interpolate_cell_to_face_2d",
    "interpolate_cell_to_face_3d",
    "interpolate_face_to_cell_1d",
    "interpolate_face_to_cell_2d",
    "interpolate_face_to_cell_3d",
    "interpolate_cell_to_node_1d",
    "interpolate_cell_to_node_2d",
    "interpolate_cell_to_node_3d",
    "interpolate_node_to_cell_1d",
    "interpolate_node_to_cell_2d",
    "interpolate_node_to_cell_3d",
]


@pytest.mark.parametrize("function_name", EXPECTED_FUNCTIONS)
def test_interpolation_is_exposed(function_name):
    assert hasattr(_mole, function_name), (
        f"_mole does not expose {function_name}"
    )
    assert callable(getattr(_mole, function_name))