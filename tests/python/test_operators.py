"""Tests for MOLE differential operators exposed to Python."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import _mole


def assert_valid_sparse_operator(operator, expected_shape):
    """Verify the basic properties of a MOLE sparse operator."""

    assert isinstance(operator, csr_matrix)
    assert operator.shape == expected_shape
    assert operator.nnz > 0
    assert np.all(np.isfinite(operator.data))

    # Input size must match the number of columns
    vector = np.ones(operator.shape[1])
    result = operator @ vector

    # Output size must match the number of rows
    assert result.shape == (operator.shape[0],)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_shape"),
    [
        (
            "laplacian_1d",
            (2, 10, 0.1),
            (12, 12),
        ),
        (
            "gradient_1d",
            (2, 10, 0.1),
            (11, 12),
        ),
        (
            "divergence_1d",
            (2, 10, 0.1),
            (12, 11),
        ),
    ],
)
def test_1d_operators(function_name, arguments, expected_shape):
    function = getattr(_mole, function_name)
    operator = function(*arguments)

    assert_valid_sparse_operator(operator, expected_shape)


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_shape"),
    [
        (
            "laplacian_2d",
            (2, 6, 5, 1.0 / 6.0, 1.0 / 5.0),
            (56, 56),
        ),
        (
            "gradient_2d",
            (2, 6, 5, 1.0 / 6.0, 1.0 / 5.0),
            (71, 56),
        ),
        (
            "divergence_2d",
            (2, 6, 5, 1.0 / 6.0, 1.0 / 5.0),
            (56, 71),
        ),
    ],
)
def test_2d_operators(function_name, arguments, expected_shape):
    function = getattr(_mole, function_name)
    operator = function(*arguments)

    assert_valid_sparse_operator(operator, expected_shape)


@pytest.mark.parametrize(
    ("function_name", "arguments", "expected_shape"),
    [
        (
            "laplacian_3d",
            (2, 5, 5, 5, 0.2, 0.2, 0.2),
            (343, 343),
        ),
        (
            "gradient_3d",
            (2, 5, 5, 5, 0.2, 0.2, 0.2),
            (450, 343),
        ),
        (
            "divergence_3d",
            (2, 5, 5, 5, 0.2, 0.2, 0.2),
            (343, 450),
        ),
    ],
)
def test_3d_operators(function_name, arguments, expected_shape):
    function = getattr(_mole, function_name)
    operator = function(*arguments)

    assert_valid_sparse_operator(operator, expected_shape)


@pytest.mark.parametrize(
    "function_name",
    [
        "laplacian_1d",
        "laplacian_2d",
        "laplacian_3d",
        "gradient_1d",
        "gradient_2d",
        "gradient_3d",
        "divergence_1d",
        "divergence_2d",
        "divergence_3d",
    ],
)
def test_operator_is_exposed(function_name):
    """Every expected operator must be present in the Python module."""

    assert hasattr(_mole, function_name), (
        f"_mole does not expose {function_name}"
    )
    assert callable(getattr(_mole, function_name))

ORDERS = [2, 4, 6, 8]
NULLITY_TOLERANCE = 1e-10


@pytest.mark.parametrize("order", ORDERS)
def test_divergence_nullity(order):
    """Replicate tests/cpp/test1.cpp: DivergenceTests.Nullity."""

    cells = 2 * order + 1
    dx = 1.0

    divergence = _mole.divergence_1d(order, cells, dx)
    constant_face_field = np.ones(cells + 1)

    result = divergence @ constant_face_field

    assert np.linalg.norm(result) < NULLITY_TOLERANCE


@pytest.mark.parametrize("order", ORDERS)
def test_gradient_nullity(order):
    """Replicate tests/cpp/test2.cpp: GradientTests.Nullity."""

    cells = 2 * order + 1
    dx = 1.0

    gradient = _mole.gradient_1d(order, cells, dx)
    constant_scalar_field = np.ones(cells + 2)

    result = gradient @ constant_scalar_field

    assert np.linalg.norm(result) < NULLITY_TOLERANCE


@pytest.mark.parametrize("order", ORDERS)
def test_laplacian_nullity(order):
    """Replicate tests/cpp/test3.cpp: LaplacianTests.Nullity."""

    cells = 2 * order + 1
    dx = 1.0

    laplacian = _mole.laplacian_1d(order, cells, dx)
    constant_scalar_field = np.ones(cells + 2)

    result = laplacian @ constant_scalar_field

    assert np.linalg.norm(result) < NULLITY_TOLERANCE