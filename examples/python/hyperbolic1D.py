"""
Solve the 1D linear advection equation:

    u_t + a*u_x = 0

using MOLE spatial operators and Leapfrog time integration.
"""

import numpy as np
import mole

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    # Parameters
    a = 1.0
    west = 0.0
    east = 1.0

    k = 2
    m = 50

    dx = (east - west) / m
    tf = 1.0
    dt = dx / abs(a)

    # MOLE staggered grid
    grid = np.zeros(m + 2)
    grid[0] = west

    for j in range(1, m + 1):
        grid[j] = west + dx / 2.0 + (j - 1) * dx

    grid[m + 1] = east

    # Initial condition
    U = np.sin(2.0 * np.pi * grid)

    # MOLE operators
    D = mole.divergence_1d(k, m, dx)
    I = mole.interpolation_1d(m, 0.5)

    # LIL format permits efficient matrix-entry modification
    D = D.tolil()

    # Periodic boundary conditions
    D[0, 1] = 1.0 / (2.0 * dx)
    D[0, -2] = -1.0 / (2.0 * dx)

    D[-1, 1] = 1.0 / (2.0 * dx)
    D[-1, -2] = -1.0 / (2.0 * dx)

    D = D.tocsr()

    # Leapfrog update operator
    Adv = -a * dt * 2.0 * (D @ I)

    # Number of time steps
    steps = round(tf / dt)

    # Leapfrog requires two time levels
    Uold = U.copy()

    # Explicit Euler startup step
    Ucur = Uold + 0.5 * (Adv @ Uold)

    numerical_history = []
    exact_history = []

    with open("build/hyperbolic1D_python_results.dat", "w") as output:
        for n in range(1, steps + 1):
            t = n * dt
            u_exact = np.sin(2.0 * np.pi * (grid - a * t))

            numerical_history.append(Ucur.copy())
            exact_history.append(u_exact.copy())

            for x, numerical, exact in zip(grid, Ucur, u_exact):
                output.write(
                    f"{x:.17e} {numerical:.17e} {exact:.17e}\n"
                )

            output.write("\n\n")

            # Leapfrog step
            Unext = Uold + Adv @ Ucur

            Uold = Ucur
            Ucur = Unext

    numerical_history = np.asarray(numerical_history)
    exact_history = np.asarray(exact_history)

    difference = numerical_history - exact_history


    # ---------------------------------------------------------
    # Animated plot
    # ---------------------------------------------------------
    numerical_history = np.asarray(numerical_history)
    exact_history = np.asarray(exact_history)

    difference = numerical_history - exact_history

    fig, ax = plt.subplots(figsize=(9, 5))

    numerical_line, = ax.plot(
        grid,
        numerical_history[0],
        "o-",
        markersize=4,
        linewidth=1.5,
        label="Numerical solution",
    )

    exact_line, = ax.plot(
        grid,
        exact_history[0],
        "-",
        linewidth=2,
        label="Exact solution",
    )

    ax.set_xlim(west, east)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"1D Advection Equation — t = {dt:.2f}")
    ax.grid(True)
    ax.legend()


    def update(frame):
        numerical_line.set_ydata(numerical_history[frame])
        exact_line.set_ydata(exact_history[frame])

        time = (frame + 1) * dt
        ax.set_title(f"1D Advection Equation — t = {time:.2f}")

        return numerical_line, exact_line


    animation = FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=100,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()

    print("Hyperbolic 1D simulation completed.")
    print(f"Number of time steps: {steps}")
    print(f"Final time: {steps * dt:.16e}")
    print(
        "Final maximum error: "
        f"{np.max(np.abs(difference[-1])):.16e}"
    )
    print(
        "Global maximum error: "
        f"{np.max(np.abs(difference)):.16e}"
    )
    print(
        "Results saved to: "
        "build/hyperbolic1D_python_results.dat"
    )


if __name__ == "__main__":
    main()