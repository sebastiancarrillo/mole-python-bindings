"""Numerical-accuracy tests translated from the MOLE C++ suite."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

import _mole


def staggered_grid(west, east, cells):
    """Construct the 1D staggered grid used by MOLE."""

    dx = (east - west) / cells

    grid = np.empty(cells + 2)
    grid[0] = west
    grid[1] = west + dx / 2.0

    for index in range(2, cells + 1):
        grid[index] = grid[index - 1] + dx

    grid[cells + 1] = east

    return grid


@pytest.mark.parametrize("order", [2, 4, 6])
def test_poisson_convergence_order(order):
    """
    Replicate tests/cpp/test5.cpp.

    Solve:

        u'' = exp(x)

    with boundary conditions matching the exact solution:

        u(x) = exp(x)
    """

    west = 0.0
    east = 1.0
    grid_sizes = [20, 40]
    errors = []

    for cells in grid_sizes:
        dx = (east - west) / cells

        laplacian = _mole.laplacian_1d(
            order,
            cells,
            dx,
        )

        boundary_operator = _mole.robin_bc_1d(
            order,
            cells,
            dx,
            1.0,
            1.0,
        )

        system_matrix = laplacian + boundary_operator

        grid = staggered_grid(west, east, cells)

        rhs = np.exp(grid)
        rhs[0] = 0.0
        rhs[-1] = 2.0 * np.exp(1.0)

        computed_solution = spsolve(
            system_matrix.tocsc(),
            rhs,
        )

        analytical_solution = np.exp(grid)

        maximum_error = np.max(
            np.abs(computed_solution - analytical_solution)
        )
        errors.append(maximum_error)

    observed_order = np.log2(errors[0] / errors[1])

    print(
        f"k={order}, "
        f"errors={errors}, "
        f"observed order={observed_order}"
    )

    assert observed_order >= order - 0.5

def test_harmonic_oscillator_eigenvalues():
    """
    Replicate tests/cpp/test4.cpp.

    Verify the first normalized energy levels of the harmonic oscillator.
    """

    order = 4
    west = -5.0
    east = 5.0
    number_of_points = 500
    tolerance = 1e-10

    grid = np.linspace(
        west,
        east,
        number_of_points,
    )
    dx = grid[1] - grid[0]

    laplacian = _mole.laplacian_1d(
        order,
        number_of_points - 2,
        dx,
    )

    potential = np.diag(grid**2)

    hamiltonian = (
        -0.5 * laplacian.toarray()
        + potential
    )

    eigenvalues = np.linalg.eigvals(hamiltonian)

    # Armadillo sorts complex eigenvalues according to magnitude
    eigenvalues = eigenvalues[
        np.argsort(np.abs(eigenvalues))
    ]

    expected_ratios = np.array(
        [1.0, 3.0, 5.0, 7.0, 9.0]
    )

    computed_ratios = np.real(
        eigenvalues[:5] / eigenvalues[0]
    )

    print("First five eigenvalues:")
    print(eigenvalues[:5])

    print("Computed ratios:")
    print(computed_ratios)

    print("Expected ratios:")
    print(expected_ratios)

    for computed, expected in zip(
        computed_ratios,
        expected_ratios,
    ):
        # Exact equivalent of the C++ expression:
        # std::norm(real(eigval(i)/eigval(0)) - expected(i))
        squared_error = (computed - expected) ** 2

        assert squared_error < tolerance