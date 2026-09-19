/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Python bindings for MOLE AddScalarBC functions.
 */

#include "common.h"

namespace {

using FloatArray = py::array_t<
    Real,
    py::array::c_style | py::array::forcecast
>;


/**
 * Validate the sizes of the boundary coefficient arrays.
 */
void check_coefficients(
    const FloatArray& dirichlet,
    const FloatArray& neumann,
    py::ssize_t expected_size
)
{
    if (dirichlet.ndim() != 1 || neumann.ndim() != 1) {
        throw py::value_error(
            "Boundary coefficients must be one-dimensional arrays."
        );
    }

    if (
        dirichlet.shape(0) != expected_size ||
        neumann.shape(0) != expected_size
    ) {
        throw py::value_error(
            "Boundary coefficient arrays have an invalid size."
        );
    }
}


/**
 * Confirm that A and b represent a compatible square system.
 */
void check_system_dimensions(
    const sp_mat& matrix,
    const vec& rhs
)
{
    if (matrix.n_rows != matrix.n_cols) {
        throw py::value_error(
            "The system matrix must be square."
        );
    }

    if (matrix.n_rows != rhs.n_elem) {
        throw py::value_error(
            "The matrix and right-hand-side dimensions do not match."
        );
    }
}


/**
 * Validate the number of boundary-value vectors.
 */
void check_boundary_values(
    const py::sequence& values,
    py::ssize_t expected_size
)
{
    if (values.size() != expected_size) {
        throw py::value_error(
            "The number of boundary-value arrays is invalid."
        );
    }
}

} // anonymous namespace


void bind_add_scalar_bc(py::module_& module)
{
    using mole_python::to_arma_spmat;
    using mole_python::to_arma_vec;
    using mole_python::to_arma_vector_of_vecs;
    using mole_python::to_numpy;
    using mole_python::to_scipy_csr;

    // ================================================================
    // 1-D scalar boundary conditions
    // ================================================================

    module.def(
        "add_scalar_bc_1d",
        [](
            const py::object& input_matrix,
            const FloatArray& input_rhs,
            u16 order,
            u32 cells,
            Real dx,
            const FloatArray& dirichlet,
            const FloatArray& neumann,
            const FloatArray& values
        ) {
            check_coefficients(dirichlet, neumann, 2);

            if (values.ndim() != 1 || values.shape(0) != 2) {
                throw py::value_error(
                    "The 1-D boundary values must contain "
                    "[left, right]."
                );
            }

            sp_mat matrix = to_arma_spmat(input_matrix);
            vec rhs = to_arma_vec(input_rhs);

            check_system_dimensions(matrix, rhs);

            AddScalarBC::BC1D boundary;
            boundary.dc = to_arma_vec(dirichlet);
            boundary.nc = to_arma_vec(neumann);
            boundary.v = to_arma_vec(values);

            AddScalarBC::addScalarBC(
                matrix,
                rhs,
                order,
                cells,
                dx,
                boundary
            );

            return py::make_tuple(
                to_scipy_csr(matrix),
                to_numpy(rhs)
            );
        },
        py::arg("matrix"),
        py::arg("rhs"),
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        py::arg("values"),
        R"doc(
Apply scalar boundary conditions to a 1-D linear system.

Parameters
----------
matrix
    Square SciPy sparse matrix.
rhs
    One-dimensional NumPy right-hand side.
order
    Mimetic accuracy order.
cells
    Number of cells.
dx
    Cell spacing.
dirichlet
    Two coefficients ordered as [left, right].
neumann
    Two coefficients ordered as [left, right].
values
    Two boundary values ordered as [left, right].

Returns
-------
matrix, rhs
    Modified SciPy CSR matrix and NumPy array.
)doc"
    );

    // ================================================================
    // 2-D scalar boundary conditions
    // ================================================================

    module.def(
        "add_scalar_bc_2d",
        [](
            const py::object& input_matrix,
            const FloatArray& input_rhs,
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy,
            const FloatArray& dirichlet,
            const FloatArray& neumann,
            const py::sequence& values
        ) {
            check_coefficients(dirichlet, neumann, 4);
            check_boundary_values(values, 4);

            sp_mat matrix = to_arma_spmat(input_matrix);
            vec rhs = to_arma_vec(input_rhs);

            check_system_dimensions(matrix, rhs);

            AddScalarBC::BC2D boundary;
            boundary.dc = to_arma_vec(dirichlet);
            boundary.nc = to_arma_vec(neumann);
            boundary.v = to_arma_vector_of_vecs(values);

            AddScalarBC::addScalarBC(
                matrix,
                rhs,
                order,
                cells_x,
                dx,
                cells_y,
                dy,
                boundary
            );

            return py::make_tuple(
                to_scipy_csr(matrix),
                to_numpy(rhs)
            );
        },
        py::arg("matrix"),
        py::arg("rhs"),
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        py::arg("values"),
        R"doc(
Apply scalar boundary conditions to a 2-D linear system.

Boundary order:

    left, right, bottom, top

`values` must be a sequence containing four one-dimensional arrays.

Returns the modified SciPy CSR matrix and NumPy RHS.
)doc"
    );

    // ================================================================
    // 3-D scalar boundary conditions
    // ================================================================

    module.def(
        "add_scalar_bc_3d",
        [](
            const py::object& input_matrix,
            const FloatArray& input_rhs,
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz,
            const FloatArray& dirichlet,
            const FloatArray& neumann,
            const py::sequence& values
        ) {
            check_coefficients(dirichlet, neumann, 6);
            check_boundary_values(values, 6);

            sp_mat matrix = to_arma_spmat(input_matrix);
            vec rhs = to_arma_vec(input_rhs);

            check_system_dimensions(matrix, rhs);

            AddScalarBC::BC3D boundary;
            boundary.dc = to_arma_vec(dirichlet);
            boundary.nc = to_arma_vec(neumann);
            boundary.v = to_arma_vector_of_vecs(values);

            AddScalarBC::addScalarBC(
                matrix,
                rhs,
                order,
                cells_x,
                dx,
                cells_y,
                dy,
                cells_z,
                dz,
                boundary
            );

            return py::make_tuple(
                to_scipy_csr(matrix),
                to_numpy(rhs)
            );
        },
        py::arg("matrix"),
        py::arg("rhs"),
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dz"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        py::arg("values"),
        R"doc(
Apply scalar boundary conditions to a 3-D linear system.

Boundary order:

    left, right, bottom, top, front, back

`values` must be a sequence containing six one-dimensional arrays.

Returns the modified SciPy CSR matrix and NumPy RHS.
)doc"
    );
}