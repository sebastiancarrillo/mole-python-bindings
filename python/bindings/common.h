/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Common conversion utilities for the MOLE Python interface.
 */

#ifndef MOLE_PYTHON_COMMON_H
#define MOLE_PYTHON_COMMON_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mole.h"

#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace mole_python {

/**
 * Convert an Armadillo sparse matrix into a SciPy CSR matrix.
 */
inline py::object to_scipy_csr(const sp_mat& matrix)
{
    const arma::uword nonzeros = matrix.n_nonzero;

    py::array_t<arma::uword> rows(nonzeros);
    py::array_t<arma::uword> columns(nonzeros);
    py::array_t<Real> values(nonzeros);

    auto rows_view = rows.mutable_unchecked<1>();
    auto columns_view = columns.mutable_unchecked<1>();
    auto values_view = values.mutable_unchecked<1>();

    arma::uword index = 0;

    for (auto entry = matrix.begin();
         entry != matrix.end();
         ++entry) {
        rows_view(index) = entry.row();
        columns_view(index) = entry.col();
        values_view(index) = *entry;
        ++index;
    }

    py::module_ sparse = py::module_::import("scipy.sparse");

    py::object coo_matrix = sparse.attr("coo_matrix")(
        py::make_tuple(
            values,
            py::make_tuple(rows, columns)
        ),
        py::arg("shape") = py::make_tuple(
            matrix.n_rows,
            matrix.n_cols
        )
    );

    return coo_matrix.attr("tocsr")();
}


/**
 * Convert an Armadillo floating-point vector into a NumPy array.
 */
inline py::array_t<Real> to_numpy(const vec& vector)
{
    py::array_t<Real> result(vector.n_elem);
    auto result_view = result.mutable_unchecked<1>();

    for (arma::uword index = 0;
         index < vector.n_elem;
         ++index) {
        result_view(index) = vector(index);
    }

    return result;
}


/**
 * Convert a one-dimensional NumPy floating-point array into arma::vec.
 */
inline vec to_arma_vec(
    const py::array_t<
        Real,
        py::array::c_style | py::array::forcecast
    >& input
)
{
    if (input.ndim() != 1) {
        throw py::value_error(
            "Expected a one-dimensional floating-point array."
        );
    }

    auto input_view = input.unchecked<1>();
    vec result(input.shape(0));

    for (py::ssize_t index = 0;
         index < input.shape(0);
         ++index) {
        result(static_cast<arma::uword>(index)) =
            input_view(index);
    }

    return result;
}


/**
 * Convert a one-dimensional NumPy integer array into arma::ivec.
 */
inline ivec to_arma_ivec(
    const py::array_t<
        long long,
        py::array::c_style | py::array::forcecast
    >& input
)
{
    if (input.ndim() != 1) {
        throw py::value_error(
            "Expected a one-dimensional integer array."
        );
    }

    auto input_view = input.unchecked<1>();
    ivec result(input.shape(0));

    for (py::ssize_t index = 0;
         index < input.shape(0);
         ++index) {
        result(static_cast<arma::uword>(index)) =
            static_cast<arma::sword>(input_view(index));
    }

    return result;
}


/**
 * Convert a one-dimensional NumPy integer array into arma::uvec.
 */
inline uvec to_arma_uvec(
    const py::array_t<
        unsigned long long,
        py::array::c_style | py::array::forcecast
    >& input
)
{
    if (input.ndim() != 1) {
        throw py::value_error(
            "Expected a one-dimensional nonnegative integer array."
        );
    }

    auto input_view = input.unchecked<1>();
    uvec result(input.shape(0));

    for (py::ssize_t index = 0;
         index < input.shape(0);
         ++index) {
        result(static_cast<arma::uword>(index)) =
            static_cast<arma::uword>(input_view(index));
    }

    return result;
}


/**
 * Convert a Python sequence of one-dimensional arrays into
 * std::vector<arma::vec>.
 */
inline std::vector<vec> to_arma_vector_of_vecs(
    const py::sequence& sequence
)
{
    std::vector<vec> result;
    result.reserve(sequence.size());

    for (const py::handle item : sequence) {
        auto array = py::array_t<
            Real,
            py::array::c_style | py::array::forcecast
        >::ensure(item);

        if (!array) {
            throw py::type_error(
                "Every boundary value must be convertible "
                "to a NumPy floating-point array."
            );
        }

        result.push_back(to_arma_vec(array));
    }

    return result;
}


/**
 * Convert a SciPy sparse matrix into an Armadillo sparse matrix.
 *
 * The input is converted to SciPy COO format before being copied.
 */
inline sp_mat to_arma_spmat(const py::object& input)
{
    if (!py::hasattr(input, "tocoo")) {
        throw py::type_error(
            "Expected a SciPy sparse matrix."
        );
    }

    py::object coo = input.attr("tocoo")();
    py::tuple shape = coo.attr("shape").cast<py::tuple>();

    const arma::uword rows_count =
        shape[0].cast<arma::uword>();

    const arma::uword columns_count =
        shape[1].cast<arma::uword>();

    auto rows = coo.attr("row").cast<
        py::array_t<
            arma::uword,
            py::array::c_style | py::array::forcecast
        >
    >();

    auto columns = coo.attr("col").cast<
        py::array_t<
            arma::uword,
            py::array::c_style | py::array::forcecast
        >
    >();

    auto values = coo.attr("data").cast<
        py::array_t<
            Real,
            py::array::c_style | py::array::forcecast
        >
    >();

    auto rows_view = rows.unchecked<1>();
    auto columns_view = columns.unchecked<1>();
    auto values_view = values.unchecked<1>();

    if (
        rows.shape(0) != columns.shape(0) ||
        rows.shape(0) != values.shape(0)
    ) {
        throw py::value_error(
            "Invalid SciPy COO matrix representation."
        );
    }

    sp_mat result(rows_count, columns_count);

    for (py::ssize_t index = 0;
         index < values.shape(0);
         ++index) {
        result(
            rows_view(index),
            columns_view(index)
        ) = values_view(index);
    }

    return result;
}

} // namespace mole_python

#endif // MOLE_PYTHON_COMMON_H