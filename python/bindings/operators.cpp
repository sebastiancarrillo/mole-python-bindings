/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Python bindings for MOLE differential operators.
 */

#include "common.h"

namespace {

using IntArray = py::array_t<
    long long,
    py::array::c_style | py::array::forcecast
>;


/**
 * Verify the required number of boundary coefficients.
 */
void check_boundary_size(
    const IntArray& dirichlet,
    const IntArray& neumann,
    py::ssize_t expected_size
)
{
    if (
        dirichlet.ndim() != 1 ||
        neumann.ndim() != 1 ||
        dirichlet.shape(0) != expected_size ||
        neumann.shape(0) != expected_size
    ) {
        throw py::value_error(
            "Boundary coefficient arrays have an invalid size."
        );
    }
}

} // anonymous namespace


void bind_operators(py::module_& module)
{
    using mole_python::to_arma_ivec;
    using mole_python::to_numpy;
    using mole_python::to_scipy_csr;

    // ================================================================
    // Laplacian
    // ================================================================

    module.def(
        "laplacian_1d",
        [](u16 order, u32 cells, Real dx) {
            Laplacian matrix(order, cells, dx);
            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        "Construct a 1-D mimetic Laplacian."
    );

    module.def(
        "laplacian_2d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy
        ) {
            Laplacian matrix(
                order,
                cells_x,
                cells_y,
                dx,
                dy
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        "Construct a 2-D mimetic Laplacian."
    );

    module.def(
        "laplacian_3d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz
        ) {
            Laplacian matrix(
                order,
                cells_x,
                cells_y,
                cells_z,
                dx,
                dy,
                dz
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dz"),
        "Construct a 3-D mimetic Laplacian."
    );

    // ================================================================
    // Gradient: non-periodic constructors
    // ================================================================

    module.def(
        "gradient_1d",
        [](u16 order, u32 cells, Real dx) {
            Gradient matrix(order, cells, dx);
            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        "Construct a non-periodic 1-D mimetic gradient."
    );

    module.def(
        "gradient_2d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy
        ) {
            Gradient matrix(
                order,
                cells_x,
                cells_y,
                dx,
                dy
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        "Construct a non-periodic 2-D mimetic gradient."
    );

    module.def(
        "gradient_3d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz
        ) {
            Gradient matrix(
                order,
                cells_x,
                cells_y,
                cells_z,
                dx,
                dy,
                dz
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dz"),
        "Construct a non-periodic 3-D mimetic gradient."
    );

    // ================================================================
    // Gradient: boundary-aware and periodic constructors
    // ================================================================

    module.def(
        "gradient_1d_bc",
        [](
            u16 order,
            u32 cells,
            Real dx,
            const IntArray& dirichlet,
            const IntArray& neumann
        ) {
            check_boundary_size(dirichlet, neumann, 2);

            Gradient matrix(
                order,
                cells,
                dx,
                to_arma_ivec(dirichlet),
                to_arma_ivec(neumann)
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        R"doc(
Construct a boundary-aware 1-D gradient.

All-zero Dirichlet and Neumann arrays create a periodic operator.
The arrays must contain [left, right].
)doc"
    );

    module.def(
        "gradient_2d_bc",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy,
            const IntArray& dirichlet,
            const IntArray& neumann
        ) {
            check_boundary_size(dirichlet, neumann, 4);

            Gradient matrix(
                order,
                cells_x,
                cells_y,
                dx,
                dy,
                to_arma_ivec(dirichlet),
                to_arma_ivec(neumann)
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        R"doc(
Construct a boundary-aware 2-D gradient.

Boundary order: left, right, bottom, top.
An axis is periodic when its corresponding coefficients are zero.
)doc"
    );

    module.def(
        "gradient_3d_bc",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz,
            const IntArray& dirichlet,
            const IntArray& neumann
        ) {
            check_boundary_size(dirichlet, neumann, 6);

            Gradient matrix(
                order,
                cells_x,
                cells_y,
                cells_z,
                dx,
                dy,
                dz,
                to_arma_ivec(dirichlet),
                to_arma_ivec(neumann)
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dz"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        R"doc(
Construct a boundary-aware 3-D gradient.

Boundary order: left, right, bottom, top, front, back.
An axis is periodic when its corresponding coefficients are zero.
)doc"
    );

    // ================================================================
    // Divergence: non-periodic constructors
    // ================================================================

    module.def(
        "divergence_1d",
        [](u16 order, u32 cells, Real dx) {
            Divergence matrix(order, cells, dx);
            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        "Construct a non-periodic 1-D mimetic divergence."
    );

    module.def(
        "divergence_2d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy
        ) {
            Divergence matrix(
                order,
                cells_x,
                cells_y,
                dx,
                dy
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        "Construct a non-periodic 2-D mimetic divergence."
    );

    module.def(
        "divergence_3d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz
        ) {
            Divergence matrix(
                order,
                cells_x,
                cells_y,
                cells_z,
                dx,
                dy,
                dz
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dz"),
        "Construct a non-periodic 3-D mimetic divergence."
    );

    // ================================================================
    // Divergence: boundary-aware and periodic constructors
    // ================================================================

    module.def(
        "divergence_1d_bc",
        [](
            u16 order,
            u32 cells,
            Real dx,
            const IntArray& dirichlet,
            const IntArray& neumann
        ) {
            check_boundary_size(dirichlet, neumann, 2);

            Divergence matrix(
                order,
                cells,
                dx,
                to_arma_ivec(dirichlet),
                to_arma_ivec(neumann)
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        R"doc(
Construct a boundary-aware 1-D divergence.

All-zero Dirichlet and Neumann arrays create a periodic operator.
The arrays must contain [left, right].
)doc"
    );

    module.def(
        "divergence_2d_bc",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy,
            const IntArray& dirichlet,
            const IntArray& neumann
        ) {
            check_boundary_size(dirichlet, neumann, 4);

            Divergence matrix(
                order,
                cells_x,
                cells_y,
                dx,
                dy,
                to_arma_ivec(dirichlet),
                to_arma_ivec(neumann)
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        R"doc(
Construct a boundary-aware 2-D divergence.

Boundary order: left, right, bottom, top.
An axis is periodic when its corresponding coefficients are zero.
)doc"
    );

    module.def(
        "divergence_3d_bc",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz,
            const IntArray& dirichlet,
            const IntArray& neumann
        ) {
            check_boundary_size(dirichlet, neumann, 6);

            Divergence matrix(
                order,
                cells_x,
                cells_y,
                cells_z,
                dx,
                dy,
                dz,
                to_arma_ivec(dirichlet),
                to_arma_ivec(neumann)
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dz"),
        py::arg("dirichlet"),
        py::arg("neumann"),
        R"doc(
Construct a boundary-aware 3-D divergence.

Boundary order: left, right, bottom, top, front, back.
An axis is periodic when its corresponding coefficients are zero.
)doc"
    );

    // ================================================================
    // Mimetic weights
    // ================================================================

    module.def(
        "gradient_weights_1d",
        [](u16 order, u32 cells, Real dx) {
            Gradient matrix(order, cells, dx);
            return to_numpy(matrix.getP());
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        "Return the P weights of a non-periodic 1-D gradient."
    );

    module.def(
        "divergence_weights_1d",
        [](u16 order, u32 cells, Real dx) {
            Divergence matrix(order, cells, dx);
            return to_numpy(matrix.getQ());
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        "Return the Q weights of a non-periodic 1-D divergence."
    );
}