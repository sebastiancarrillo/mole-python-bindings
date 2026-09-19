/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Python bindings for MOLE interpolation operators.
 */

#include "common.h"

#include <string>

namespace {

using IntArray = py::array_t<
    long long,
    py::array::c_style | py::array::forcecast
>;


/**
 * Validate Dirichlet and Neumann coefficient arrays.
 */
void check_boundary_size(
    const IntArray& dirichlet,
    const IntArray& neumann,
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
 * Construct a 1-D directional interpolation operator.
 */
template <typename Operator>
py::object directional_interpolation_1d(
    u16 order,
    u32 cells,
    const IntArray& dirichlet,
    const IntArray& neumann
)
{
    check_boundary_size(dirichlet, neumann, 2);

    Operator matrix(
        order,
        cells,
        mole_python::to_arma_ivec(dirichlet),
        mole_python::to_arma_ivec(neumann)
    );

    return mole_python::to_scipy_csr(matrix);
}


/**
 * Construct a 2-D directional interpolation operator.
 */
template <typename Operator>
py::object directional_interpolation_2d(
    u16 order,
    u32 cells_x,
    u32 cells_y,
    const IntArray& dirichlet,
    const IntArray& neumann
)
{
    check_boundary_size(dirichlet, neumann, 4);

    Operator matrix(
        order,
        cells_x,
        cells_y,
        mole_python::to_arma_ivec(dirichlet),
        mole_python::to_arma_ivec(neumann)
    );

    return mole_python::to_scipy_csr(matrix);
}


/**
 * Construct a 3-D directional interpolation operator.
 */
template <typename Operator>
py::object directional_interpolation_3d(
    u16 order,
    u32 cells_x,
    u32 cells_y,
    u32 cells_z,
    const IntArray& dirichlet,
    const IntArray& neumann
)
{
    check_boundary_size(dirichlet, neumann, 6);

    Operator matrix(
        order,
        cells_x,
        cells_y,
        cells_z,
        mole_python::to_arma_ivec(dirichlet),
        mole_python::to_arma_ivec(neumann)
    );

    return mole_python::to_scipy_csr(matrix);
}


/**
 * Register the 1-D, 2-D and 3-D constructors shared by the
 * center/face/node interpolation classes.
 */
template <typename Operator>
void bind_directional_interpolator(
    py::module_& module,
    const std::string& python_name
)
{
    const std::string name_1d = python_name + "_1d";
    const std::string name_2d = python_name + "_2d";
    const std::string name_3d = python_name + "_3d";

    module.def(
        name_1d.c_str(),
        &directional_interpolation_1d<Operator>,
        py::arg("order"),
        py::arg("cells"),
        py::arg("dirichlet"),
        py::arg("neumann")
    );

    module.def(
        name_2d.c_str(),
        &directional_interpolation_2d<Operator>,
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("dirichlet"),
        py::arg("neumann")
    );

    module.def(
        name_3d.c_str(),
        &directional_interpolation_3d<Operator>,
        py::arg("order"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("dirichlet"),
        py::arg("neumann")
    );
}

} // anonymous namespace


void bind_interpolation(py::module_& module)
{
    using mole_python::to_scipy_csr;

    // ================================================================
    // General Interpol operators
    // ================================================================

    module.def(
        "interpolation_1d",
        [](u32 cells, Real weight) {
            Interpol matrix(cells, weight);
            return to_scipy_csr(matrix);
        },
        py::arg("cells"),
        py::arg("weight"),
        "Construct a 1-D mimetic interpolation operator."
    );

    module.def(
        "interpolation_2d",
        [](
            u32 cells_x,
            u32 cells_y,
            Real weight_x,
            Real weight_y
        ) {
            Interpol matrix(
                cells_x,
                cells_y,
                weight_x,
                weight_y
            );

            return to_scipy_csr(matrix);
        },
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("weight_x"),
        py::arg("weight_y"),
        "Construct a 2-D mimetic interpolation operator."
    );

    module.def(
        "interpolation_3d",
        [](
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real weight_x,
            Real weight_y,
            Real weight_z
        ) {
            Interpol matrix(
                cells_x,
                cells_y,
                cells_z,
                weight_x,
                weight_y,
                weight_z
            );

            return to_scipy_csr(matrix);
        },
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("weight_x"),
        py::arg("weight_y"),
        py::arg("weight_z"),
        "Construct a 3-D mimetic interpolation operator."
    );

    // ================================================================
    // Alternate Interpol overloads
    //
    // C++ uses the boolean only to select the alternate constructor.
    // ================================================================

    module.def(
        "interpolation_alternate_1d",
        [](bool type, u32 cells, Real weight) {
            Interpol matrix(type, cells, weight);
            return to_scipy_csr(matrix);
        },
        py::arg("type"),
        py::arg("cells"),
        py::arg("weight"),
        "Construct the alternate 1-D interpolation operator."
    );

    module.def(
        "interpolation_alternate_2d",
        [](
            bool type,
            u32 cells_x,
            u32 cells_y,
            Real weight_x,
            Real weight_y
        ) {
            Interpol matrix(
                type,
                cells_x,
                cells_y,
                weight_x,
                weight_y
            );

            return to_scipy_csr(matrix);
        },
        py::arg("type"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("weight_x"),
        py::arg("weight_y"),
        "Construct the alternate 2-D interpolation operator."
    );

    module.def(
        "interpolation_alternate_3d",
        [](
            bool type,
            u32 cells_x,
            u32 cells_y,
            u32 cells_z,
            Real weight_x,
            Real weight_y,
            Real weight_z
        ) {
            Interpol matrix(
                type,
                cells_x,
                cells_y,
                cells_z,
                weight_x,
                weight_y,
                weight_z
            );

            return to_scipy_csr(matrix);
        },
        py::arg("type"),
        py::arg("cells_x"),
        py::arg("cells_y"),
        py::arg("cells_z"),
        py::arg("weight_x"),
        py::arg("weight_y"),
        py::arg("weight_z"),
        "Construct the alternate 3-D interpolation operator."
    );

    // ================================================================
    // Specialized interpolation operators
    // ================================================================

    bind_directional_interpolator<InterpolCtoF>(
        module,
        "interpolate_cell_to_face"
    );

    bind_directional_interpolator<InterpolCtoN>(
        module,
        "interpolate_cell_to_node"
    );

    bind_directional_interpolator<InterpolFtoC>(
        module,
        "interpolate_face_to_cell"
    );

    bind_directional_interpolator<InterpolNtoC>(
        module,
        "interpolate_node_to_cell"
    );
}