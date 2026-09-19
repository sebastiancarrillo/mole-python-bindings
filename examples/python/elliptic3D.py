"""
Three-dimensional Laplace boundary-value problem.

Python equivalent of:
    examples/cpp/elliptic3D.cpp
"""

from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

import mole


def main():
    order = 2

    cells_x = 5
    cells_y = 6
    cells_z = 7

    dx = 1.0
    dy = 1.0
    dz = 1.0

    dirichlet_coefficient = 1.0
    neumann_coefficient = 0.0

    print("\n=== 3D Laplace Equation ===\n")

    # MOLE operators
    laplacian = mole.laplacian_3d(
        order=order,
        cells_x=cells_x,
        cells_y=cells_y,
        cells_z=cells_z,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    boundary = mole.robin_bc_3d(
        order=order,
        cells_x=cells_x,
        cells_y=cells_y,
        cells_z=cells_z,
        dx=dx,
        dy=dy,
        dz=dz,
        dirichlet_coefficient=dirichlet_coefficient,
        neumann_coefficient=neumann_coefficient,
    )

    system = laplacian + boundary

    shape = (
        cells_x + 2,
        cells_y + 2,
        cells_z + 2,
    )

    # Same RHS as the C++ arma::cube
    rhs_cube = np.zeros(shape, dtype=np.float64)

    # Front face z = 0
    rhs_cube[:, :, 0] = 100.0

    # Armadillo vectorise() uses column-major ordering.
    rhs = rhs_cube.ravel(order="F")

    print(f"Operator size: {system.shape}")
    print(f"Non-zero elements: {system.nnz}")
    print("Solving linear system...")

    solution = spsolve(system, rhs)

    # Reconstruct the cube using Armadillo ordering.
    solution_cube = solution.reshape(shape, order="F")

    residual = system @ solution - rhs
    residual_inf = np.linalg.norm(residual, ord=np.inf)

    print("Solution obtained")
    print(f"Minimum solution: {solution.min():.12e}")
    print(f"Maximum solution: {solution.max():.12e}")
    print(f"Residual infinity norm: {residual_inf:.12e}")

    # C++ uses p / 2 with integer division.
    middle_z = cells_z // 2
    middle_slice = solution_cube[:, :, middle_z]

    output_path = Path(
        "build/elliptic3D_python_solution.txt"
    )

    with output_path.open("w", encoding="utf-8") as output:
        for i in range(cells_x + 2):
            for j in range(cells_y + 2):
                output.write(
                    f"{i} {j} {middle_slice[i, j]:.16e}\n"
                )

            output.write("\n")

    print(f"Middle slice saved to: {output_path}")


if __name__ == "__main__":
    main()