/**
 * Benchmark large 2-D MOLE operators directly in C++.
 */

#include "mole.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


struct BenchmarkResult {
    double median_seconds;
    arma::uword rows;
    arma::uword columns;
    arma::uword nonzeros;
};


double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const std::size_t size = values.size();
    const std::size_t middle = size / 2;

    if (size % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }

    return values[middle];
}


template <typename Constructor>
BenchmarkResult benchmark(
    const std::string& name,
    Constructor constructor,
    int repeats
)
{
    std::vector<double> times;

    arma::uword rows = 0;
    arma::uword columns = 0;
    arma::uword nonzeros = 0;

    for (int repetition = 0; repetition < repeats; ++repetition) {
        const auto start = std::chrono::steady_clock::now();

        auto matrix = constructor();

        const auto finish = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = finish - start;
        times.push_back(elapsed.count());

        rows = matrix.n_rows;
        columns = matrix.n_cols;
        nonzeros = matrix.n_nonzero;

        // Read matrix information so the compiler cannot discard construction.
        volatile arma::uword observed_nonzeros = matrix.n_nonzero;
        (void) observed_nonzeros;
    }

    const double median_seconds = median(times);

    std::cout
        << std::left << std::setw(28) << name
        << " shape=(" << rows << ", " << columns << ")"
        << " nnz=" << std::setw(12) << nonzeros
        << " median=" << std::fixed << std::setprecision(6)
        << median_seconds << " s"
        << " runs=[";

    for (std::size_t index = 0; index < times.size(); ++index) {
        if (index > 0) {
            std::cout << ", ";
        }

        std::cout << std::fixed << std::setprecision(6) << times[index];
    }

    std::cout << "]\n";

    return {
        median_seconds,
        rows,
        columns,
        nonzeros
    };
}


int main(int argc, char** argv)
{
    const u32 cells =
        argc > 1 ? static_cast<u32>(std::stoul(argv[1])) : 500;

    const u16 order =
        argc > 2 ? static_cast<u16>(std::stoul(argv[2])) : 4;

    const int repeats =
        argc > 3 ? std::stoi(argv[3]) : 3;

    const Real dx = 1.0 / static_cast<Real>(cells);
    const Real dy = 1.0 / static_cast<Real>(cells);

    arma::ivec dirichlet(4, arma::fill::ones);
    arma::ivec neumann(4, arma::fill::zeros);

    std::cout << "\nMOLE C++ large 2-D benchmark\n";
    std::cout << "Order:       " << order << '\n';
    std::cout << "Cells:       " << cells << " x " << cells << '\n';
    std::cout << "Spacing:     " << dx << " x " << dy << '\n';
    std::cout << "Repetitions: " << repeats << "\n\n";

    std::cout
        << "C++ timings include operator construction only; "
        << "there is no conversion to SciPy.\n\n";

    std::vector<std::pair<std::string, double>> results;

    auto laplacian_result = benchmark(
        "laplacian_2d",
        [=]() {
            return Laplacian(
                order,
                cells,
                cells,
                dx,
                dy
            );
        },
        repeats
    );

    results.push_back({
        "laplacian_2d",
        laplacian_result.median_seconds
    });

    auto gradient_result = benchmark(
        "gradient_2d",
        [=]() {
            return Gradient(
                order,
                cells,
                cells,
                dx,
                dy
            );
        },
        repeats
    );

    results.push_back({
        "gradient_2d",
        gradient_result.median_seconds
    });

    auto divergence_result = benchmark(
        "divergence_2d",
        [=]() {
            return Divergence(
                order,
                cells,
                cells,
                dx,
                dy
            );
        },
        repeats
    );

    results.push_back({
        "divergence_2d",
        divergence_result.median_seconds
    });

    auto interpolation_result = benchmark(
        "interpolation_2d",
        [=]() {
            return Interpol(
                cells,
                cells,
                0.5,
                0.5
            );
        },
        repeats
    );

    results.push_back({
        "interpolation_2d",
        interpolation_result.median_seconds
    });

    auto cell_to_face_result = benchmark(
        "interpolate_cell_to_face_2d",
        [=]() {
            return InterpolCtoF(
                order,
                cells,
                cells,
                dirichlet,
                neumann
            );
        },
        repeats
    );

    results.push_back({
        "cell_to_face_2d",
        cell_to_face_result.median_seconds
    });

    std::cout << "\nCSV summary\n";
    std::cout << "operation,cpp_seconds\n";

    for (const auto& result : results) {
        std::cout
            << result.first << ','
            << std::fixed << std::setprecision(9)
            << result.second << '\n';
    }

    return EXIT_SUCCESS;
}