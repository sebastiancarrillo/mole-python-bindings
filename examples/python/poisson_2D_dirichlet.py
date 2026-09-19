"""
Two-dimensional Poisson equation with Dirichlet conditions.

Python equivalent of:
    examples/cpp/poisson_2D_dirichlet.cpp
"""

import numpy as np
from scipy.sparse.linalg import spsolve

import mole


def main():
    print("\n=== 2D Poisson Equation with Dirichlet BCs ===\n")

    # Parameters
    order = 2
    cells_x = 30
    cells_y = 30

    dx = 1.0 / (cells_x + 1)
    dy = 1.0 / (cells_y + 1)

    print("Grid parameters:")
    print(f"  Order of accuracy: {order}")
    print(f"  Grid size: {cells_x} x {cells_y}")
    print(f"  Cell spacing: dx = {dx}, dy = {dy}\n")

    # Construct -L because the problem is -Laplacian(u) = f
    print("Constructing 2D Laplacian operator...")

    laplacian = mole.laplacian_2d(
        order=order,
        cells_x=cells_x,
        cells_y=cells_y,
        dx=dx,
        dy=dy,
    )

    matrix = -laplacian

    print(f"  Operator size: {matrix.shape}")
    print(f"  Non-zero elements: {matrix.nnz}\n")

    # Same grid used by the C++ example
    x = np.linspace(0.0, 1.0, cells_x + 2)
    y = np.linspace(0.0, 1.0, cells_y + 2)

    # meshgrid creates arrays with shape:
    #
    #     (cells_y + 2, cells_x + 2)
    #
    # C-order flattening gives idx = j*(cells_x+2) + i,
    # matching the C++ implementation.
    X, Y = np.meshgrid(x, y, indexing="xy")

    exact_2d = np.sin(np.pi * X) * np.sin(np.pi * Y)

    rhs_2d = (
        2.0
        * np.pi**2
        * np.sin(np.pi * X)
        * np.sin(np.pi * Y)
    )

    rhs = rhs_2d.ravel(order="C")

    # Dirichlet on every boundary
    dirichlet = np.array(
        [1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    neumann = np.array(
        [0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    # Boundary order:
    # left, right, bottom, top
    boundary_values = [
        np.zeros(cells_y, dtype=np.float64),
        np.zeros(cells_y, dtype=np.float64),
        np.zeros(cells_x + 2, dtype=np.float64),
        np.zeros(cells_x + 2, dtype=np.float64),
    ]

    print("Applying Dirichlet boundary conditions...")

    matrix, rhs = mole.add_scalar_bc_2d(
        matrix=matrix,
        rhs=rhs,
        order=order,
        cells_x=cells_x,
        cells_y=cells_y,
        dx=dx,
        dy=dy,
        dirichlet=dirichlet,
        neumann=neumann,
        values=boundary_values,
    )

    print("  Boundary conditions applied")
    print(f"  Modified operator size: {matrix.shape}\n")

    # Solve A*u = f
    print("Solving linear system...")

    solution = spsolve(matrix, rhs)

    print("  Solution obtained\n")

    exact = exact_2d.ravel(order="C")

    # Error analysis
    error = solution - exact
    max_error = np.max(np.abs(error))
    l2_error = np.linalg.norm(error) / np.sqrt(solution.size)

    residual = matrix @ solution - rhs
    residual_inf = np.linalg.norm(residual, ord=np.inf)

    print("Error analysis:")
    print(f"  Max error: {max_error:.12e}")
    print(f"  L2 error:  {l2_error:.12e}")
    print(f"  Residual:  {residual_inf:.12e}\n")

    if max_error < 1.0e-2:
        print("Solution accurate within tolerance!")
    else:
        print("Warning: Error may be higher than expected")

    # Same flattened indices used in C++
    center_index = (
        (cells_y + 2) // 2 * (cells_x + 2)
        + (cells_x + 2) // 2
    )

    quarter_index = (
        (cells_y + 2) // 4 * (cells_x + 2)
        + (cells_x + 2) // 4
    )

    three_quarter_index = (
        3 * (cells_y + 2) // 4 * (cells_x + 2)
        + 3 * (cells_x + 2) // 4
    )

    print("\nSample solution values:")

    print(
        f"  Center ≈ {solution[center_index]:.12f} "
        f"(exact at grid point: {exact[center_index]:.12f})"
    )

    print(
        f"  First quarter ≈ {solution[quarter_index]:.12f} "
        f"(exact at grid point: {exact[quarter_index]:.12f})"
    )

    print(
        f"  Third quarter ≈ "
        f"{solution[three_quarter_index]:.12f} "
        f"(exact at grid point: "
        f"{exact[three_quarter_index]:.12f})"
    )


if __name__ == "__main__":
    main()