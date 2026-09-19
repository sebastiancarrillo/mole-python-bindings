"""
Solve the same 1-D boundary-value problem as
examples/cpp/elliptic1D.cpp.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

import mole


def solve_elliptic_1d():
    order = 6
    a = 0.0
    b = 1.0
    cells = 2 * order + 1
    dx = (b - a) / cells

    # Same MOLE operators used by the C++ example
    laplacian = mole.laplacian_1d(
        order=order,
        cells=cells,
        dx=dx,
    )

    boundary = mole.robin_bc_1d(
        order=order,
        cells=cells,
        dx=dx,
        dirichlet_coefficient=1.0,
        neumann_coefficient=1.0,
    )

    system = laplacian + boundary

    # MOLE staggered grid
    grid = np.empty(cells + 2)
    grid[0] = a
    grid[1] = a + dx / 2.0

    for index in range(2, cells + 1):
        grid[index] = grid[index - 1] + dx

    grid[-1] = b

    # Same RHS used by the C++ example
    rhs = np.exp(grid)
    rhs[0] = 0.0
    rhs[-1] = 2.0 * np.exp(1.0)

    solution = spsolve(system, rhs)

    return grid, solution


if __name__ == "__main__":
    grid, solution = solve_elliptic_1d()

    # Match the four-decimal output shown by the C++ example
    for value in solution:
        print(f"{value:.4f}")