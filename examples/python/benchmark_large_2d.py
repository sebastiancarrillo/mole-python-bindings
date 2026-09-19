"""Benchmark large 2-D MOLE operators through the Python interface."""

import argparse
import gc
import statistics
import time

import mole
import numpy as np
from scipy.sparse import csr_matrix


def benchmark(name, constructor, repeats):
    """Measure construction plus C++-to-SciPy conversion time."""

    times = []
    shape = None
    nonzeros = None

    for _ in range(repeats):
        gc.collect()

        start = time.perf_counter()
        matrix = constructor()
        elapsed = time.perf_counter() - start

        assert isinstance(matrix, csr_matrix)

        times.append(elapsed)
        shape = matrix.shape
        nonzeros = matrix.nnz

        # Make sure the matrix is actually usable.
        vector = np.ones(matrix.shape[1])
        result = matrix @ vector
        assert np.all(np.isfinite(result))

        del result
        del vector
        del matrix

    median = statistics.median(times)

    print(
        f"{name:28s}"
        f" shape={str(shape):20s}"
        f" nnz={nonzeros:12,d}"
        f" median={median:10.6f} s"
        f" runs={[round(value, 6) for value in times]}"
    )

    return median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=500)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    cells = arguments.cells
    order = arguments.order
    repeats = arguments.repeats

    dx = 1.0 / cells
    dy = 1.0 / cells

    dirichlet = np.ones(4, dtype=np.int64)
    neumann = np.zeros(4, dtype=np.int64)

    print("\nMOLE Python large 2-D benchmark")
    print(f"Order:      {order}")
    print(f"Cells:      {cells} x {cells}")
    print(f"Spacing:    {dx} x {dy}")
    print(f"Repetitions:{repeats}")
    print()
    print(
        "Python timings include both C++ construction and conversion "
        "to SciPy CSR.\n"
    )

    # Small warm-up calls load SciPy and initialize the extension before
    # measuring the large matrices.
    mole.laplacian_1d(2, 10, 0.1)
    mole.interpolation_1d(10, 0.5)

    results = {}

    results["laplacian_2d"] = benchmark(
        "laplacian_2d",
        lambda: mole.laplacian_2d(
            order,
            cells,
            cells,
            dx,
            dy,
        ),
        repeats,
    )

    results["gradient_2d"] = benchmark(
        "gradient_2d",
        lambda: mole.gradient_2d(
            order,
            cells,
            cells,
            dx,
            dy,
        ),
        repeats,
    )

    results["divergence_2d"] = benchmark(
        "divergence_2d",
        lambda: mole.divergence_2d(
            order,
            cells,
            cells,
            dx,
            dy,
        ),
        repeats,
    )

    results["interpolation_2d"] = benchmark(
        "interpolation_2d",
        lambda: mole.interpolation_2d(
            cells,
            cells,
            0.5,
            0.5,
        ),
        repeats,
    )

    results["cell_to_face_2d"] = benchmark(
        "interpolate_cell_to_face_2d",
        lambda: mole.interpolate_cell_to_face_2d(
            order,
            cells,
            cells,
            dirichlet,
            neumann,
        ),
        repeats,
    )

    print("\nCSV summary")
    print("operation,python_seconds")

    for name, elapsed in results.items():
        print(f"{name},{elapsed:.9f}")


if __name__ == "__main__":
    main()