"""Tests translated from tests/cpp/test_addscalarbc.cpp."""

import numpy as np
import pytest
from scipy.sparse.linalg import norm as sparse_norm

import _mole


TOLERANCE = 1e-10


def matrix_difference(matrix_a, matrix_b):
    """Frobenius norm of A - B."""

    return sparse_norm(matrix_a - matrix_b)


def vector_difference(vector_a, vector_b):
    """Euclidean norm of a - b."""

    return np.linalg.norm(vector_a - vector_b)


def row_is_identity(matrix, row, column, tolerance=TOLERANCE):
    """Check whether one sparse-matrix row is an identity constraint."""

    dense_row = matrix.getrow(row).toarray().ravel()
    expected = np.zeros(matrix.shape[1])
    expected[column] = 1.0

    return np.allclose(
        dense_row,
        expected,
        rtol=0.0,
        atol=tolerance,
    )


# ------------------------------------------------------------------
# 1D tests
# ------------------------------------------------------------------


def apply_1d_bc(dirichlet, neumann, values):
    order = 2
    cells = 10
    dx = 0.1

    original_matrix = _mole.laplacian_1d(
        order,
        cells,
        dx,
    )
    original_rhs = np.ones(cells + 2)

    modified_matrix, modified_rhs = _mole.add_scalar_bc_1d(
        original_matrix,
        original_rhs,
        order,
        cells,
        dx,
        np.asarray(dirichlet, dtype=float),
        np.asarray(neumann, dtype=float),
        np.asarray(values, dtype=float),
    )

    return (
        original_matrix,
        original_rhs,
        modified_matrix,
        modified_rhs,
    )


def test_1d_dirichlet():
    original_matrix, _, matrix, rhs = apply_1d_bc(
        dirichlet=[1.0, 1.0],
        neumann=[0.0, 0.0],
        values=[1.0, 0.0],
    )

    assert rhs[0] == pytest.approx(1.0, abs=TOLERANCE)
    assert rhs[-1] == pytest.approx(0.0, abs=TOLERANCE)
    assert matrix_difference(matrix, original_matrix) > 0.0
    assert row_is_identity(matrix, 0, 0)
    assert row_is_identity(matrix, matrix.shape[0] - 1, matrix.shape[1] - 1)


def test_1d_neumann():
    original_matrix, _, matrix, rhs = apply_1d_bc(
        dirichlet=[0.0, 0.0],
        neumann=[1.0, 1.0],
        values=[0.0, 0.0],
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert rhs[0] == pytest.approx(0.0, abs=TOLERANCE)
    assert rhs[-1] == pytest.approx(0.0, abs=TOLERANCE)
    assert not row_is_identity(matrix, 0, 0)
    assert not row_is_identity(
        matrix,
        matrix.shape[0] - 1,
        matrix.shape[1] - 1,
    )


def test_1d_dirichlet_left_neumann_right():
    original_matrix, _, matrix, rhs = apply_1d_bc(
        dirichlet=[1.0, 0.0],
        neumann=[0.0, 1.0],
        values=[2.0, 0.5],
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert rhs[0] == pytest.approx(2.0, abs=TOLERANCE)
    assert rhs[-1] == pytest.approx(0.5, abs=TOLERANCE)
    assert row_is_identity(matrix, 0, 0)
    assert not row_is_identity(
        matrix,
        matrix.shape[0] - 1,
        matrix.shape[1] - 1,
    )


def test_1d_neumann_left_dirichlet_right():
    original_matrix, _, matrix, rhs = apply_1d_bc(
        dirichlet=[0.0, 1.0],
        neumann=[1.0, 0.0],
        values=[0.0, -1.0],
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert rhs[0] == pytest.approx(0.0, abs=TOLERANCE)
    assert rhs[-1] == pytest.approx(-1.0, abs=TOLERANCE)
    assert not row_is_identity(matrix, 0, 0)
    assert row_is_identity(
        matrix,
        matrix.shape[0] - 1,
        matrix.shape[1] - 1,
    )


def test_1d_robin():
    original_matrix, _, matrix, rhs = apply_1d_bc(
        dirichlet=[2.0, 3.0],
        neumann=[1.0, 4.0],
        values=[1.5, -0.25],
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert rhs[0] == pytest.approx(1.5, abs=TOLERANCE)
    assert rhs[-1] == pytest.approx(-0.25, abs=TOLERANCE)
    assert not row_is_identity(matrix, 0, 0)
    assert not row_is_identity(
        matrix,
        matrix.shape[0] - 1,
        matrix.shape[1] - 1,
    )


def test_1d_periodic_does_nothing():
    original_matrix, original_rhs, matrix, rhs = apply_1d_bc(
        dirichlet=[0.0, 0.0],
        neumann=[0.0, 0.0],
        values=[0.0, 0.0],
    )

    assert matrix_difference(matrix, original_matrix) < TOLERANCE
    assert vector_difference(rhs, original_rhs) < TOLERANCE


# ------------------------------------------------------------------
# 2D tests
# ------------------------------------------------------------------


def apply_2d_bc(dirichlet, neumann, values, cells_x=8, cells_y=7):
    order = 2
    dx = 0.1
    dy = 0.2

    size = (cells_x + 2) * (cells_y + 2)

    original_matrix = _mole.laplacian_2d(
        order,
        cells_x,
        cells_y,
        dx,
        dy,
    )
    original_rhs = np.ones(size)

    modified_matrix, modified_rhs = _mole.add_scalar_bc_2d(
        original_matrix,
        original_rhs,
        order,
        cells_x,
        cells_y,
        dx,
        dy,
        np.asarray(dirichlet, dtype=float),
        np.asarray(neumann, dtype=float),
        [np.asarray(value, dtype=float) for value in values],
    )

    return (
        original_matrix,
        original_rhs,
        modified_matrix,
        modified_rhs,
    )


def test_2d_dirichlet_all_boundaries():
    cells_x = 8
    cells_y = 7

    values = [
        np.full(cells_y + 2, 1.0),
        np.full(cells_y + 2, 2.0),
        np.full(cells_x + 2, 3.0),
        np.full(cells_x + 2, 4.0),
    ]

    original_matrix, original_rhs, matrix, rhs = apply_2d_bc(
        dirichlet=[1.0, 1.0, 1.0, 1.0],
        neumann=[0.0, 0.0, 0.0, 0.0],
        values=values,
        cells_x=cells_x,
        cells_y=cells_y,
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert vector_difference(rhs, original_rhs) > 0.0


def test_2d_neumann_all_boundaries():
    cells_x = 8
    cells_y = 7

    values = [
        np.zeros(cells_y + 2),
        np.zeros(cells_y + 2),
        np.zeros(cells_x + 2),
        np.zeros(cells_x + 2),
    ]

    original_matrix, original_rhs, matrix, rhs = apply_2d_bc(
        dirichlet=[0.0, 0.0, 0.0, 0.0],
        neumann=[1.0, 1.0, 1.0, 1.0],
        values=values,
        cells_x=cells_x,
        cells_y=cells_y,
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert vector_difference(rhs, original_rhs) > 0.0


def test_2d_mixed_boundaries():
    cells_x = 10
    cells_y = 10

    values = [
        np.full(cells_y + 2, 1.0),
        np.zeros(cells_y + 2),
        np.zeros(cells_x + 2),
        np.zeros(cells_x + 2),
    ]

    original_matrix, original_rhs, matrix, rhs = apply_2d_bc(
        dirichlet=[1.0, 1.0, 0.0, 0.0],
        neumann=[0.0, 0.0, 1.0, 1.0],
        values=values,
        cells_x=cells_x,
        cells_y=cells_y,
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert vector_difference(rhs, original_rhs) > 0.0


# ------------------------------------------------------------------
# 3D tests
# ------------------------------------------------------------------


def apply_3d_bc(dirichlet, neumann, values):
    order = 2
    cells_x = 6
    cells_y = 6
    cells_z = 6

    dx = 0.1
    dy = 0.1
    dz = 0.1

    size = (
        (cells_x + 2)
        * (cells_y + 2)
        * (cells_z + 2)
    )

    original_matrix = _mole.laplacian_3d(
        order,
        cells_x,
        cells_y,
        cells_z,
        dx,
        dy,
        dz,
    )
    original_rhs = np.ones(size)

    modified_matrix, modified_rhs = _mole.add_scalar_bc_3d(
        original_matrix,
        original_rhs,
        order,
        cells_x,
        cells_y,
        cells_z,
        dx,
        dy,
        dz,
        np.asarray(dirichlet, dtype=float),
        np.asarray(neumann, dtype=float),
        [np.asarray(value, dtype=float) for value in values],
    )

    return (
        original_matrix,
        original_rhs,
        modified_matrix,
        modified_rhs,
    )


def make_3d_values(left=0.0, right=0.0):
    cells_x = 6
    cells_y = 6
    cells_z = 6

    return [
        np.full((cells_y + 2) * (cells_z + 2), left),
        np.full((cells_y + 2) * (cells_z + 2), right),
        np.zeros((cells_x + 2) * (cells_z + 2)),
        np.zeros((cells_x + 2) * (cells_z + 2)),
        np.zeros((cells_x + 2) * (cells_y + 2)),
        np.zeros((cells_x + 2) * (cells_y + 2)),
    ]


def test_3d_dirichlet_all_boundaries():
    original_matrix, original_rhs, matrix, rhs = apply_3d_bc(
        dirichlet=[1.0] * 6,
        neumann=[0.0] * 6,
        values=make_3d_values(left=1.0, right=0.0),
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert vector_difference(rhs, original_rhs) > 0.0


def test_3d_neumann_all_boundaries():
    original_matrix, original_rhs, matrix, rhs = apply_3d_bc(
        dirichlet=[0.0] * 6,
        neumann=[1.0] * 6,
        values=make_3d_values(),
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert vector_difference(rhs, original_rhs) > 0.0


def test_3d_mixed_boundaries():
    original_matrix, original_rhs, matrix, rhs = apply_3d_bc(
        dirichlet=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        neumann=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        values=make_3d_values(left=1.0, right=2.0),
    )

    assert matrix_difference(matrix, original_matrix) > 0.0
    assert vector_difference(rhs, original_rhs) > 0.0