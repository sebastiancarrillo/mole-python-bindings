"""
One-dimensional advection-reaction-dispersion problem.

Python equivalent of:
    examples/cpp/transport1D.cpp
"""

import numpy as np

import mole


def main():
    order = 2
    left = 0.0
    right = 130.0
    cells = 26
    dx = (right - left) / cells

    final_time = 4.0
    iterations = 208
    dt = final_time / iterations

    dispersivity = 5.0
    velocity = 15.0
    retardation = 2.5
    initial_concentration = 1.0

    # Construct the same MOLE operators as C++
    gradient = mole.gradient_1d(
        order=order,
        cells=cells,
        dx=dx,
    )

    divergence = mole.divergence_1d(
        order=order,
        cells=cells,
        dx=dx,
    )

    interpolation = mole.interpolation_1d(
        cells=cells,
        weight=0.5,
    )

    # Scalar and vector fields
    concentration = np.zeros(
        cells + 2,
        dtype=np.float64,
    )

    velocities = np.full(
        cells + 1,
        velocity,
        dtype=np.float64,
    )

    concentration[0] = initial_concentration

    # Hydrodynamic dispersion coefficient
    dispersion = dispersivity * velocity

    # Retardation modifies the time step
    dt /= retardation

    # The C++ example uses i <= iterations,
    # therefore it performs iterations + 1 updates.
    for _ in range(iterations + 1):
        concentration += dt * (
            divergence @ (
                dispersion * (gradient @ concentration)
            )
            - divergence @ (
                velocities * (interpolation @ concentration)
            )
        )

        # Reflecting right boundary
        concentration[-1] = concentration[-2]

    for value in concentration:
        print(f"{value:.4e}")


if __name__ == "__main__":
    main()