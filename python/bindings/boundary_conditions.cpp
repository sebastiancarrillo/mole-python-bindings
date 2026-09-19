/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Python bindings for MOLE boundary-condition operators.
 */

#include "common.h"

#include <string>
#include <vector>


void bind_boundary_conditions(py::module_& module)
{
    using mole_python::to_scipy_csr;

    // ================================================================
    // Robin boundary conditions
    // ================================================================

    module.def(
        "robin_bc_1d",
        [](
            u16 order,
            u32 cells,
            Real dx,
            Real dirichlet_coefficient,
            Real neumann_coefficient
        ) {
            RobinBC matrix(
                order,
                cells,
                dx,
                dirichlet_coefficient,
                neumann_coefficient
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        py::arg("dirichlet_coefficient"),
        py::arg("neumann_coefficient"),
        R"doc(
Construct a 1-D Robin boundary-condition operator.

The boundary equation has the form:

    a*u + b*du/dn = g

where `dirichlet_coefficient` is a and
`neumann_coefficient` is b.
)doc"
    );

    module.def(
        "robin_bc_2d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy,
            Real dirichlet_coefficient,
            Real neumann_coefficient
        ) {
            RobinBC matrix(
                order,
                cells_x,
                dx,
                cells_y,
                dy,
                dirichlet_coefficient,
                neumann_coefficient
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("dirichlet_coefficient"),
        py::arg("neumann_coefficient"),
        "Construct a 2-D Robin boundary-condition operator."
    );

    module.def(
        "robin_bc_3d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz,
            Real dirichlet_coefficient,
            Real neumann_coefficient
        ) {
            RobinBC matrix(
                order,
                cells_x,
                dx,
                cells_y,
                dy,
                cells_z,
                dz,
                dirichlet_coefficient,
                neumann_coefficient
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
        py::arg("dirichlet_coefficient"),
        py::arg("neumann_coefficient"),
        "Construct a 3-D Robin boundary-condition operator."
    );

    // ================================================================
    // Mixed boundary conditions
    // ================================================================

    module.def(
        "mixed_bc_1d",
        [](
            u16 order,
            u32 cells,
            Real dx,
            const std::string& left_type,
            const std::vector<Real>& left_coefficients,
            const std::string& right_type,
            const std::vector<Real>& right_coefficients
        ) {
            MixedBC matrix(
                order,
                cells,
                dx,
                left_type,
                left_coefficients,
                right_type,
                right_coefficients
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells"),
        py::arg("dx"),
        py::arg("left_type"),
        py::arg("left_coefficients"),
        py::arg("right_type"),
        py::arg("right_coefficients"),
        R"doc(
Construct a 1-D mixed boundary-condition operator.

Valid boundary types are:

    "Dirichlet"
    "Neumann"
    "Robin"
)doc"
    );

    module.def(
        "mixed_bc_2d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            Real dx,
            Real dy,
            const std::string& left_type,
            const std::vector<Real>& left_coefficients,
            const std::string& right_type,
            const std::vector<Real>& right_coefficients,
            const std::string& bottom_type,
            const std::vector<Real>& bottom_coefficients,
            const std::string& top_type,
            const std::vector<Real>& top_coefficients
        ) {
            MixedBC matrix(
                order,
                cells_x,
                dx,
                cells_y,
                dy,
                left_type,
                left_coefficients,
                right_type,
                right_coefficients,
                bottom_type,
                bottom_coefficients,
                top_type,
                top_coefficients
            );

            return to_scipy_csr(matrix);
        },
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dx"),
        py::arg("dy"),
        py::arg("left_type"),
        py::arg("left_coefficients"),
        py::arg("right_type"),
        py::arg("right_coefficients"),
        py::arg("bottom_type"),
        py::arg("bottom_coefficients"),
        py::arg("top_type"),
        py::arg("top_coefficients"),
        R"doc(
Construct a 2-D mixed boundary-condition operator.

Boundary order:

    left, right, bottom, top
)doc"
    );

    module.def(
        "mixed_bc_3d",
        [](
            u16 order,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real dx,
            Real dy,
            Real dz,
            const std::string& left_type,
            const std::vector<Real>& left_coefficients,
            const std::string& right_type,
            const std::vector<Real>& right_coefficients,
            const std::string& bottom_type,
            const std::vector<Real>& bottom_coefficients,
            const std::string& top_type,
            const std::vector<Real>& top_coefficients,
            const std::string& front_type,
            const std::vector<Real>& front_coefficients,
            const std::string& back_type,
            const std::vector<Real>& back_coefficients
        ) {
            MixedBC matrix(
                order,
                cells_x,
                dx,
                cells_y,
                dy,
                cells_z,
                dz,
                left_type,
                left_coefficients,
                right_type,
                right_coefficients,
                bottom_type,
                bottom_coefficients,
                top_type,
                top_coefficients,
                front_type,
                front_coefficients,
                back_type,
                back_coefficients
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
        py::arg("left_type"),
        py::arg("left_coefficients"),
        py::arg("right_type"),
        py::arg("right_coefficients"),
        py::arg("bottom_type"),
        py::arg("bottom_coefficients"),
        py::arg("top_type"),
        py::arg("top_coefficients"),
        py::arg("front_type"),
        py::arg("front_coefficients"),
        py::arg("back_type"),
        py::arg("back_coefficients"),
        R"doc(
Construct a 3-D mixed boundary-condition operator.

Boundary order:

    left, right, bottom, top, front, back
)doc"
    );
}