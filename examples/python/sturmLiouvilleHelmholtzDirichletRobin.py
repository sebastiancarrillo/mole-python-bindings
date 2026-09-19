"""
Solve the 1D Helmholtz equation:

    u'' + mu^2 u = 0,    x in [0, 1]

Boundary conditions:

    u'(0) = 0
    u(1) + u'(1) = cos(mu) - mu*sin(mu)

Exact solution:

    u(x) = cos(mu*x)
"""

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

import mole


def main():
    # Parameters
    k = 2
    m = 150
    dx = 1.0 / m
    mu = 0.86

    # MOLE staggered grid
    xc = np.empty(m + 2, dtype=float)
    xc[0] = 0.0
    xc[1] = dx / 2.0

    for i in range(2, m + 1):
        xc[i] = xc[i - 1] + dx

    xc[m + 1] = 1.0

    # Exact solution
    u_exact = np.cos(mu * xc)

    # Mimetic Laplacian
    L = mole.laplacian_1d(k, m, dx)

    # Left boundary: Neumann
    # Right boundary: Robin
    mixed_bc = mole.mixed_bc_1d(
        k,
        m,
        dx,
        "Neumann",
        [1.0, 0.0],
        "Robin",
        [1.0, 1.0],
    )

    # A = L + mu^2 I
    A = L + mu**2 * eye(m + 2, format="csr")

    # Clear first and last rows before adding the boundary operator
    A = A.tolil()
    A[0, :] = 0.0
    A[-1, :] = 0.0
    A = A.tocsr()

    A = A + mixed_bc

    # Right-hand side
    b = np.zeros(m + 2, dtype=float)
    b[0] = 0.0
    b[-1] = np.cos(mu) - mu * np.sin(mu)

    # Solve sparse system
    solution = spsolve(A.tocsc(), b)

    # Error analysis
    difference = solution - u_exact

    print(f"norm(u_numerical) = {np.linalg.norm(solution):.16e}")
    print(f"norm(u_exact) = {np.linalg.norm(u_exact):.16e}")
    print(
        "norm(u_numerical - u_exact) = "
        f"{np.linalg.norm(difference):.16e}"
    )
    print(f"maximum absolute error = {np.max(np.abs(difference)):.16e}")

    # Save values for comparison with C++
    output = np.column_stack((xc, solution, u_exact, difference))
    np.savetxt(
        "build/sturmLiouville_python_solution.txt",
        output,
        header="x numerical exact difference",
        fmt="%.17e",
    )


if __name__ == "__main__":
    main()