"""Tests for MOLE boundary-aware differential operators."""

import numpy as np
import pytest
from scipy.sparse import issparse

import _mole


TOLERANCE = 1e-10


def assert_valid_sparse_matrix(matrix):
    assert issparse(matrix)
    assert matrix.ndim == 2
    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0
    assert matrix.nnz > 0
    assert np.all(np.isfinite(matrix.data))

    vector = np.ones(matrix.shape[1])
    result = matrix @ vector

    assert result.shape == (matrix.shape[0],)
    assert np.all(np.isfinite(result))


# ------------------------------------------------------------------
# Boundary-aware gradient and divergence
# ------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_periodic_gradient_1d_nullity(order):
    """Replicate the periodic test from tests/cpp/test2.cpp."""

    cells = 2 * order + 1
    coefficients = np.array([0, 0], dtype=np.int64)

    gradient = _mole.gradient_1d_bc(
        order,
        cells,
        1.0,
        coefficients,
        coefficients,
    )

    assert gradient.shape == (cells, cells)

    result = gradient @ np.ones(cells)

    assert np.linalg.norm(result) < TOLERANCE


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_periodic_divergence_1d_nullity(order):
    """Replicate the periodic test from tests/cpp/test1.cpp."""

    cells = 2 * order + 1
    coefficients = np.array([0, 0], dtype=np.int64)

    divergence = _mole.divergence_1d_bc(
        order,
        cells,
        1.0,
        coefficients,
        coefficients,
    )

    assert divergence.shape == (cells, cells)

    result = divergence @ np.ones(cells)

    assert np.linalg.norm(result) < TOLERANCE


@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        (
            "gradient_1d_bc",
            (
                2,
                10,
                0.1,
                np.array([1, 1], dtype=np.int64),
                np.array([0, 0], dtype=np.int64),
            ),
        ),
        (
            "divergence_1d_bc",
            (
                2,
                10,
                0.1,
                np.array([1, 1], dtype=np.int64),
                np.array([0, 0], dtype=np.int64),
            ),
        ),
        (
            "gradient_2d_bc",
            (
                2,
                6,
                5,
                0.1,
                0.2,
                np.array([1, 1, 1, 1], dtype=np.int64),
                np.array([0, 0, 0, 0], dtype=np.int64),
            ),
        ),
        (
            "divergence_2d_bc",
            (
                2,
                6,
                5,
                0.1,
                0.2,
                np.array([1, 1, 1, 1], dtype=np.int64),
                np.array([0, 0, 0, 0], dtype=np.int64),
            ),
        ),
        (
            "gradient_3d_bc",
            (
                2,
                5,
                5,
                5,
                0.1,
                0.2,
                0.3,
                np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
                np.array([0, 0, 0, 0, 0, 0], dtype=np.int64),
            ),
        ),
        (
            "divergence_3d_bc",
            (
                2,
                5,
                5,
                5,
                0.1,
                0.2,
                0.3,
                np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
                np.array([0, 0, 0, 0, 0, 0], dtype=np.int64),
            ),
        ),
    ],
)
def test_boundary_aware_operator(function_name, arguments):
    operator = getattr(_mole, function_name)(*arguments)

    assert_valid_sparse_matrix(operator)


# ------------------------------------------------------------------
# Robin boundary operators
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_size"),
    [
        (
            "robin_bc_1d",
            (2, 10, 0.1, 1.0, 1.0),
            12,
        ),
        (
            "robin_bc_2d",
            (2, 6, 5, 0.1, 0.2, 1.0, 1.0),
            (6 + 2) * (5 + 2),
        ),
        (
            "robin_bc_3d",
            (2, 5, 5, 5, 0.1, 0.2, 0.3, 1.0, 1.0),
            (5 + 2) * (5 + 2) * (5 + 2),
        ),
    ],
)
def test_robin_boundary_operator(
    function_name,
    arguments,
    expected_size,
):
    operator = getattr(_mole, function_name)(*arguments)

    assert_valid_sparse_matrix(operator)
    assert operator.shape == (expected_size, expected_size)


# ------------------------------------------------------------------
# Mixed boundary operators
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_size"),
    [
        (
            "mixed_bc_1d",
            (
                2,
                10,
                0.1,
                "Dirichlet",
                [1.0],
                "Robin",
                [1.0, 1.0],
            ),
            12,
        ),
        (
            "mixed_bc_2d",
            (
                2,
                6,
                5,
                0.1,
                0.2,
                "Dirichlet",
                [1.0],
                "Robin",
                [1.0, 1.0],
                "Neumann",
                [1.0],
                "Dirichlet",
                [1.0],
            ),
            (6 + 2) * (5 + 2),
        ),
        (
            "mixed_bc_3d",
            (
                2,
                5,
                5,
                5,
                0.1,
                0.2,
                0.3,
                "Dirichlet",
                [1.0],
                "Robin",
                [1.0, 1.0],
                "Neumann",
                [1.0],
                "Dirichlet",
                [1.0],
                "Robin",
                [2.0, 1.0],
                "Neumann",
                [1.0],
            ),
            (5 + 2) * (5 + 2) * (5 + 2),
        ),
    ],
)
def test_mixed_boundary_operator(
    function_name,
    arguments,
    expected_size,
):
    operator = getattr(_mole, function_name)(*arguments)

    assert_valid_sparse_matrix(operator)
    assert operator.shape == (expected_size, expected_size)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


EXPECTED_FUNCTIONS = [
    "gradient_1d_bc",
    "gradient_2d_bc",
    "gradient_3d_bc",
    "divergence_1d_bc",
    "divergence_2d_bc",
    "divergence_3d_bc",
    "robin_bc_1d",
    "robin_bc_2d",
    "robin_bc_3d",
    "mixed_bc_1d",
    "mixed_bc_2d",
    "mixed_bc_3d",
]


@pytest.mark.parametrize("function_name", EXPECTED_FUNCTIONS)
def test_boundary_function_is_exposed(function_name):
    assert hasattr(_mole, function_name)
    assert callable(getattr(_mole, function_name))